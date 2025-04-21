import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, ResBlock3d, SPADEResnetBlock, ContextHourglass, coords_grid
from modules.dense_motion import DenseMotionInit, DenseMotionNetwork, DenseMotionNetworkUpdater, UpdateBlock3D, Args
from modules.corr import CorrBlock3D
from torch.amp import autocast


class OcclusionAwareGenerator(nn.Module):
    """
    Generator for motion refinement and occlusion aware generation.
    """

    def __init__(self, image_channel, feature_channel, num_kp, block_expansion, max_features, num_down_blocks, reshape_channel, reshape_depth,
                 num_resblocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(OcclusionAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_initalizer = DenseMotionNetworkUpdater(num_kp=num_kp, feature_channel=feature_channel,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        
        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        in_features = [] # for upblocks
        out_features = [] # for upblocks
        self.num_down_blocks = num_down_blocks
        for i in range(num_down_blocks):
            in_feature = min(max_features, block_expansion * (2 ** i))
            out_feature = min(max_features, block_expansion * (2 ** (i + 1)))
            in_features.append(out_feature)
            out_features.append(in_feature)
            down_blocks.append(DownBlock2d(in_feature, out_feature, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        in_features = in_features[::-1] # Reverse the order for upblocks
        out_features = out_features[::-1] # Reverse the order for upblocks
        for i in range(num_down_blocks):
            up_blocks.append(UpBlock2d(in_features[i], out_features[i], kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)
        
        resblock = []
        for i in range(num_down_blocks):
            resblock.append(ResBlock2d(in_features[i], kernel_size=(3, 3), padding=(1, 1)))
            resblock.append(ResBlock2d(in_features[i], kernel_size=(3, 3), padding=(1, 1)))
        self.resblock = nn.ModuleList(resblock)

        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth
        
        self.resblocks_3d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(reshape_channel, kernel_size=3, padding=1))
        

        self.predict_image = nn.ModuleList()
        for in_channel in [512, 256, 128, 64]:
            self.predict_image.append(self._buil_predictor(in_channel))

        self.estimate_occlusion_map = estimate_occlusion_map
        self.image_channel = image_channel

    def _buil_predictor(self, in_channels, image_channel=3):
        return nn.Sequential(*[
                nn.Conv2d(in_channels, image_channel, kernel_size=(7, 7), padding=(3, 3)),
                nn.Sigmoid()
                ])

    def deform_input(self, inp, deformation):
        _, d_old, h_old, w_old, _ = deformation.shape
        _, _, d, h, w = inp.shape
        if d_old != d or h_old != h or w_old != w:
            deformation = deformation.permute(0, 4, 1, 2, 3) # (bs, d, h, w, 3) -> (bs, 3, d, h, w)
            deformation = F.interpolate(deformation, size=(d, h, w), mode='trilinear')
            deformation = deformation.permute(0, 2, 3, 4, 1) # (bs, 3, d, h, w) -> (bs, d, h, w, 3)
        return F.grid_sample(inp, deformation)
    
    def up_motion(self, motion, mode='trilinear'): 
        return  2 * F.interpolate(motion, size=(2 * motion.shape[2], 2 * motion.shape[3], motion.shape[4]),
                                  mode=mode, align_corners=True)

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, D, H, W = img.shape
        coords0 = coords_grid(N, D, H, W, device=img.device)
        coords1 = coords_grid(N, D, H, W, device=img.device)
        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
    
    def occlude_input(self, inp, occlusion_map, inverse=False):
        if inp.shape[2] != occlusion_map.shape[2] or inp.shape[3] != occlusion_map.shape[3]:
            occlusion_map = F.interpolate(occlusion_map, size=inp.shape[2:], mode='bilinear',align_corners=True)
        if inverse:
            occlusion_map = 1 - occlusion_map
        out = inp * occlusion_map
        return out
    
    def forward(self, source_image, driving_image, kp_driving, kp_source, rec_driving=False):
        output_dict = {}

        # Encoding (downsampling) part
        out = self.first(source_image)
        encoder_map = [out]

        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            encoder_map.append(out)
        encoder_map = list(reversed(encoder_map))
        bs, c, h, w = out.shape
        
        feature_3d = out.view(bs, self.reshape_channel, self.reshape_depth, h, w)
        feature_3d = self.resblocks_3d(feature_3d)

        # Transforming feature representation according to deformation and occlusion
        dense_motion_init = self.dense_motion_initalizer(feature=feature_3d, kp_driving=kp_driving,
                                                    kp_source=kp_source)

        if 'mask' in dense_motion_init:
            output_dict['mask'] = dense_motion_init['mask']
        motion = dense_motion_init['deformation']
        occlusion_map = dense_motion_init['occlusion_map']
        output_dict.update({'sparse_motion': dense_motion_init['sparse_motion'], 'heatmap': dense_motion_init['heatmap']})
                
        feature_3d_deformed = self.deform_input(feature_3d, motion)
        bs, c, d, h, w = feature_3d_deformed.shape
        out = feature_3d_deformed.view(bs, c*d, h, w)
        out = self.occlude_input(out, occlusion_map, inverse=False)
        
        prediction = self.predict_image[0](out)
        output_dict['deformation'] = [motion]
        output_dict['occlusion_map'] = [occlusion_map]
        output_dict["prediction"] = [prediction]

        for i in range(len(self.up_blocks)):
            # Up blocks
            out = self.resblock[2*i](out)
            out = self.resblock[2*i+1](out)
            encoder_out = self.occlude_input(encoder_map[i], occlusion_map, inverse=True)
            out = self.up_blocks[i](out+encoder_out)
            out = self.occlude_input(out, occlusion_map, inverse=False)
            
            # Predict image at scale
            prediction = self.predict_image[i+1](out)
            output_dict["prediction"].append(prediction)


        if rec_driving:
            #TODO: Adapt this to the new encoders and decoders
            out = self.first(driving_image)
            NotImplementedError("This part is not implemented yet")
        else:
            output_dict["driving_rec"] = None

        return output_dict

class OcclusionAwareGeneratorRAFT(nn.Module):
    """
    Generator follows NVIDIA architecture.
    """

    def __init__(self, image_channel, feature_channel, num_kp, block_expansion, max_features, num_down_blocks, reshape_channel, reshape_depth,
                 num_resblocks, estimate_occlusion_map=False, dense_motion_params=None, motion_refiner_params=None, estimate_jacobian=False):
        super(OcclusionAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_initalizer = DenseMotionInit(num_kp=num_kp, feature_channel=feature_channel,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        if motion_refiner_params is not None:
            self.args = Args(**motion_refiner_params)
            self.cnet = ContextHourglass(**motion_refiner_params['context_params'], out_features=self.args.hidden_dim+self.args.context_dim)
            self.motion_refiner = UpdateBlock3D(args=self.args, hidden_dim=self.args.hidden_dim)
        else:
            ValueError("No motion refinement parameter given")
        
        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        in_features = [] # for upblocks
        out_features = [] # for upblocks
        self.num_down_blocks = num_down_blocks
        for i in range(num_down_blocks):
            in_feature = min(max_features, block_expansion * (2 ** i))
            out_feature = min(max_features, block_expansion * (2 ** (i + 1)))
            in_features.append(out_feature)
            out_features.append(in_feature)
            down_blocks.append(DownBlock2d(in_feature, out_feature, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        in_features = in_features[::-1] # Reverse the order for upblocks
        out_features = out_features[::-1] # Reverse the order for upblocks
        for i in range(num_down_blocks):
            up_blocks.append(UpBlock2d(in_features[i], out_features[i], kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)
        
        resblock = []
        for i in range(num_down_blocks):
            resblock.append(ResBlock2d(in_features[i], kernel_size=(3, 3), padding=(1, 1)))
            resblock.append(ResBlock2d(in_features[i], kernel_size=(3, 3), padding=(1, 1)))
        self.resblock = nn.ModuleList(resblock)

        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth
        
        self.resblocks_3d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(reshape_channel, kernel_size=3, padding=1))
        

        self.predict_image = nn.ModuleList()
        for in_channel in [512, 256, 128, 64]:
            self.predict_image.append(self._buil_predictor(in_channel))

        self.estimate_occlusion_map = estimate_occlusion_map
        self.image_channel = image_channel

    def _buil_predictor(self, in_channels, image_channel=3):
        return nn.Sequential(*[
                nn.Conv2d(in_channels, image_channel, kernel_size=(7, 7), padding=(3, 3)),
                nn.Sigmoid()
                ])

    def deform_input(self, inp, deformation):
        _, d_old, h_old, w_old, _ = deformation.shape
        _, _, d, h, w = inp.shape
        if d_old != d or h_old != h or w_old != w:
            deformation = deformation.permute(0, 4, 1, 2, 3) # (bs, d, h, w, 3) -> (bs, 3, d, h, w)
            deformation = F.interpolate(deformation, size=(d, h, w), mode='trilinear')
            deformation = deformation.permute(0, 2, 3, 4, 1) # (bs, 3, d, h, w) -> (bs, d, h, w, 3)
        return F.grid_sample(inp, deformation)
    
    def up_motion(self, motion, mode='trilinear'): 
        return  2 * F.interpolate(motion, size=(2 * motion.shape[2], 2 * motion.shape[3], motion.shape[4]),
                                  mode=mode, align_corners=True)

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, D, H, W = img.shape
        coords0 = coords_grid(N, D, H, W, device=img.device)
        coords1 = coords_grid(N, D, H, W, device=img.device)
        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
    
    def occlude_input(self, inp, occlusion_map):
        if inp.shape[2] != occlusion_map.shape[2] or inp.shape[3] != occlusion_map.shape[3]:
            occlusion_map = F.interpolate(occlusion_map, size=inp.shape[2:], mode='bilinear',align_corners=True)
        out = inp * occlusion_map
        return out
    
    def forward(self, source_image, driving_image, kp_driving, kp_source, fmap_source, fmap_driving, rec_driving=False, iters=12):
        output_dict = {}

        # Encoding (downsampling) part
        out = self.first(source_image)
        # encoder_map = [out]

        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
            # encoder_map.append(out)
        bs, c, h, w = out.shape
        
        feature_3d = out.view(bs, self.reshape_channel, self.reshape_depth, h, w)
        feature_3d = self.resblocks_3d(feature_3d)

        # Transforming feature representation according to deformation and occlusion
        dense_motion_init = self.dense_motion_initalizer(feature=feature_3d, kp_driving=kp_driving,
                                                    kp_source=kp_source)
        # coords0, coords1 = self.initialize_flow(feature_3d)
        # occlusion_map = torch.zeros_like(source_image[:, :1, :, :])

        # output_dict['mask'] = dense_motion['mask']
        motion = dense_motion_init['deformation']
        occlusion_map = dense_motion_init['occlusion_map']
        output_dict['deformation'] = [motion]
        output_dict['occlusion_map'] = [occlusion_map]
        
        # Compute Correlation Volume Pyramid
        corr_fn = CorrBlock3D(fmap1=fmap_source, fmap2=fmap_driving,
                              radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        
        # Extract Context Features
        cnet = self.cnet(source_image)
        hdim, cdim = self.args.hidden_dim, self.args.context_dim
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        
        # Refine Motion
        motion = motion.permute(0,4,1,2,3)
        for _ in range(iters):
            # coords1 = coords1.detach()
            corr = corr_fn(motion) # index correlation volume

            # motion = coords1 - coords0
            net, delta_flow = self.motion_refiner(net, inp, corr, motion)

            # F(t+1) = F(t) + \Delta(t)
            motion = motion + delta_flow

            
            # motion = coords1 - coords0
            output_dict['deformation'].append(motion.permute(0,2,3,4,1))
        
        
        motion = motion.permute(0,2,3,4,1)
        feature_3d_deformed = self.deform_input(feature_3d, motion)
        bs, c, d, h, w = feature_3d_deformed.shape
        out = feature_3d_deformed.view(bs, c*d, h, w)
        out = self.occlude_input(out, occlusion_map)
        
        prediction = self.predict_image[0](out)
        output_dict['deformation'] = [motion]
        output_dict['occlusion_map'] = [occlusion_map]
        output_dict["prediction"] = [prediction]

        for i in range(len(self.up_blocks)):
            # Up blocks
            out = self.resblock[2*i](out)
            out = self.resblock[2*i+1](out)
            out = self.occlude_input(out, occlusion_map)
            out = self.up_blocks[i](out)
            
            # Predict image at scale
            prediction = self.predict_image[i+1](out)
            output_dict["prediction"].append(prediction)


        if rec_driving:
            #TODO: Adapt this to the new encoders and decoders
            out = self.first(driving_image)
            NotImplementedError("This part is not implemented yet")
        else:
            output_dict["driving_rec"] = None

        return output_dict


class SPADEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        ic = 256
        oc = 64
        norm_G = 'spadespectralinstance'
        label_nc = 256
        
        self.fc = nn.Conv2d(ic, 2 * ic, 3, padding=1)
        self.G_middle_0 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_1 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_2 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_3 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_4 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_5 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.up_0 = SPADEResnetBlock(2 * ic, ic, norm_G, label_nc)
        self.up_1 = SPADEResnetBlock(ic, oc, norm_G, label_nc)
        self.conv_img = nn.Conv2d(oc, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        
    def forward(self, feature):
        seg = feature
        x = self.fc(feature)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.G_middle_2(x, seg)
        x = self.G_middle_3(x, seg)
        x = self.G_middle_4(x, seg)
        x = self.G_middle_5(x, seg)
        x = self.up(x)                
        x = self.up_0(x, seg)         # 256, 128, 128
        x = self.up(x)                
        x = self.up_1(x, seg)         # 64, 256, 256

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        # x = torch.tanh(x)
        x = F.sigmoid(x)
        
        return x


class OcclusionAwareSPADEGenerator(nn.Module):

    def __init__(self, image_channel, feature_channel, num_kp, block_expansion, max_features, num_down_blocks, reshape_channel, reshape_depth,
                 num_resblocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(OcclusionAwareSPADEGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, feature_channel=feature_channel,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(3, 3), padding=(1, 1))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        self.second = nn.Conv2d(in_channels=out_features, out_channels=max_features, kernel_size=1, stride=1)

        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth

        self.resblocks_3d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(reshape_channel, kernel_size=3, padding=1))

        out_features = block_expansion * (2 ** (num_down_blocks))
        self.third = SameBlock2d(max_features, out_features, kernel_size=(3, 3), padding=(1, 1), lrelu=True)
        self.fourth = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1)

        self.estimate_occlusion_map = estimate_occlusion_map
        self.image_channel = image_channel

        self.decoder = SPADEDecoder()

    def deform_input(self, inp, deformation):
        _, d_old, h_old, w_old, _ = deformation.shape
        _, _, d, h, w = inp.shape
        if d_old != d or h_old != h or w_old != w:
            deformation = deformation.permute(0, 4, 1, 2, 3)
            deformation = F.interpolate(deformation, size=(d, h, w), mode='trilinear')
            deformation = deformation.permute(0, 2, 3, 4, 1)
        return F.grid_sample(inp, deformation)

    def forward(self, source_image, kp_driving, kp_source):
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        out = self.second(out)
        bs, c, h, w = out.shape
        # print(out.shape)
        feature_3d = out.view(bs, self.reshape_channel, self.reshape_depth, h ,w) 
        feature_3d = self.resblocks_3d(feature_3d)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(feature=feature_3d, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out = self.deform_input(feature_3d, deformation)

            bs, c, d, h, w = out.shape
            out = out.view(bs, c*d, h, w)
            out = self.third(out)
            out = self.fourth(out)

            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                out = out * occlusion_map

        # Decoding part
        out = self.decoder(out)

        output_dict["prediction"] = out

        return output_dict