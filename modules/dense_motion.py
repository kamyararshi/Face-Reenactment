from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, SPADEResnetBlock3D, make_coordinate_grid, kp2gaussian
from modules.util import mesh_grid, norm_grid

from sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d


class DenseMotionNetwork(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, feature_channel, reshape_depth, compress,
                 estimate_occlusion_map=False):
        super(DenseMotionNetwork, self).__init__()
        # self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp+1)*(feature_channel+1), max_features=max_features, num_blocks=num_blocks)
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp+1)*(compress+1), max_features=max_features, num_blocks=num_blocks)

        self.mask = nn.Conv3d(self.hourglass.out_filters, num_kp + 1, kernel_size=7, padding=3)

        self.compress = nn.Conv3d(feature_channel, compress, kernel_size=1)
        self.norm = BatchNorm3d(compress, affine=True)

        if estimate_occlusion_map:
            # self.occlusion = nn.Conv2d(reshape_channel*reshape_depth, 1, kernel_size=7, padding=3)
            self.occlusion = nn.Conv2d(self.hourglass.out_filters*reshape_depth, 1, kernel_size=7, padding=3)
        else:
            self.occlusion = None

        self.num_kp = num_kp


    def create_sparse_motions(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape
        identity_grid = make_coordinate_grid((d, h, w), type=kp_source['value'].type(), device=feature.device)
        identity_grid = identity_grid.view(1, 1, d, h, w, 3)
        coordinate_grid = identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 1, 3)
        
        k = coordinate_grid.shape[1]

        driving_to_source = coordinate_grid + kp_source['value'].view(bs, self.num_kp, 1, 1, 1, 3)    # (bs, num_kp, d, h, w, 3)

        #adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1, 1) # (bs, 1, d, h, w, 3) NOTE: repeat only the batch dimension
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1) # [(bs, 1, d, h, w, 3), (bs, num_kp, d, h, w, 3)] -> (bs, num_kp+1, d, h, w, 3)
        
        # sparse_motions = driving_to_source

        return sparse_motions

    def create_deformed_feature(self, feature, sparse_motions):
        bs, _, d, h, w = feature.shape
        feature_repeat = feature.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp+1, 1, 1, 1, 1, 1)      # (bs, num_kp+1, 1, c, d, h, w)
        feature_repeat = feature_repeat.view(bs * (self.num_kp+1), -1, d, h, w)                         # (bs*(num_kp+1), c, d, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp+1), d, h, w, -1))                       # (bs*(num_kp+1), d, h, w, 3)
        sparse_deformed = F.grid_sample(feature_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp+1, -1, d, h, w))                        # (bs, num_kp+1, c, d, h, w)
        return sparse_deformed

    def create_heatmap_representations(self, feature, kp_driving, kp_source):
        spatial_size = feature.shape[3:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=0.01)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=0.01)
        heatmap = gaussian_driving - gaussian_source

        # adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1], spatial_size[2], device=heatmap.device).type(heatmap.type())
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)         # (bs, num_kp+1, 1, d, h, w)
        return heatmap

    def forward(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape

        feature = self.compress(feature)
        feature = self.norm(feature)
        feature = F.relu(feature)

        out_dict = dict()
        sparse_motion = self.create_sparse_motions(feature, kp_driving, kp_source)
        deformed_feature = self.create_deformed_feature(feature, sparse_motion) #NOTE: Warps the feature with the sparse_motion

        heatmap = self.create_heatmap_representations(deformed_feature, kp_driving, kp_source)

        input = torch.cat([heatmap, deformed_feature], dim=2)
        input = input.view(bs, -1, d, h, w)

        # input = deformed_feature.view(bs, -1, d, h, w)      # (bs, num_kp+1 * c, d, h, w)

        prediction = self.hourglass(input) #NOTE: An encoder-decoder network

        mask = self.mask(prediction) #TODO: A 3D convolutional layer that outputs the mask
        mask = F.softmax(mask, dim=1)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)                                   # (bs, num_kp+1, 1, d, h, w)
        sparse_motion = sparse_motion.permute(0, 1, 5, 2, 3, 4)    # (bs, num_kp+1, 3, d, h, w)
        deformation = (sparse_motion * mask).sum(dim=1)            # (bs, 3, d, h, w)
        deformation = deformation.permute(0, 2, 3, 4, 1)           # (bs, d, h, w, 3)

        out_dict['deformation'] = deformation

        if self.occlusion:
            bs, c, d, h, w = prediction.shape
            prediction = prediction.view(bs, -1, h, w)
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map

        return out_dict

class DenseMotionInit(nn.Module):
    """
    Module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_kp, feature_channel, reshape_depth, compress,
                 estimate_occlusion_map=False):
        super(DenseMotionInit, self).__init__()

        self.compress = nn.Conv3d(feature_channel, compress, kernel_size=1)
        self.norm = BatchNorm3d(compress, affine=True)

        if estimate_occlusion_map:
            # self.occlusion = nn.Conv2d(compress*(num_kp+1)*reshape_depth, 1, kernel_size=7, padding=3)
            self.occlusion = nn.Conv2d(compress*reshape_depth, 1, kernel_size=7, padding=3)
        else:
            self.occlusion = None

        self.num_kp = num_kp


    def create_sparse_motions(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape
        identity_grid = make_coordinate_grid((d, h, w), type=kp_source['value'].type(), device=feature.device)
        identity_grid = identity_grid.view(1, 1, d, h, w, 3)
        coordinate_grid = identity_grid - kp_driving['value'].view(bs, self.num_kp, 1, 1, 1, 3)

        driving_to_source = coordinate_grid + kp_source['value'].view(bs, self.num_kp, 1, 1, 1, 3)    # (bs, num_kp, d, h, w, 3)

        #adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1, 1) # (bs, 1, d, h, w, 3) NOTE: repeat only the batch dimension
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1) # [(bs, 1, d, h, w, 3), (bs, num_kp, d, h, w, 3)] -> (bs, num_kp+1, d, h, w, 3)
        
        # sparse_motions = driving_to_source

        return sparse_motions

    def create_deformed_feature(self, feature, sparse_motions):
        bs, _, d, h, w = feature.shape
        feature_repeat = feature.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp+1, 1, 1, 1, 1, 1)      # (bs, num_kp+1, 1, c, d, h, w)
        feature_repeat = feature_repeat.view(bs * (self.num_kp+1), -1, d, h, w)                         # (bs*(num_kp+1), c, d, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp+1), d, h, w, -1))                       # (bs*(num_kp+1), d, h, w, 3)
        sparse_deformed = F.grid_sample(feature_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp+1, -1, d, h, w))                        # (bs, num_kp+1, c, d, h, w)
        return sparse_deformed

    def create_heatmap_representations(self, feature, kp_driving, kp_source):
        spatial_size = feature.shape[3:]
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=0.1)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=0.1)
        heatmap = gaussian_driving - gaussian_source

        # adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1], spatial_size[2], device=heatmap.device).type(heatmap.type())
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)         # (bs, num_kp+1, 1, d, h, w)
        return heatmap

    def forward(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape
        feature = self.compress(feature)
        feature = self.norm(feature)
        feature = F.relu(feature)

        out_dict = dict()
        sparse_motion = self.create_sparse_motions(feature, kp_driving, kp_source)
        # deformed_feature = self.create_deformed_feature(feature, sparse_motion) #NOTE: Warps the feature with the sparse_motion
        # deformed_feature = deformed_feature.contiguous().view(bs, -1, h, w)

        # heatmap = self.create_heatmap_representations(deformed_feature, kp_driving, kp_source)

        # heatmap shape (bs, num_kp+1, 1, d, h, w)
        sparse_motion = sparse_motion.permute(0, 1, 5, 2, 3, 4)    # (bs, num_kp+1, 3, d, h, w)
        # deformation = (sparse_motion * heatmap).sum(dim=1)            # (bs, 3, d, h, w)
        deformation = sparse_motion.mean(dim=1)            # (bs, 3, d, h, w)
        deformation = deformation.permute(0, 2, 3, 4, 1)           # (bs, d, h, w, 3)

        out_dict['deformation'] = deformation

        if self.occlusion:
            occlusion_map = torch.sigmoid(self.occlusion(feature.view(bs,-1,h,w)))
            occlusion_map = torch.ones(bs, 1, h, w).to(feature.device)
            out_dict['occlusion_map'] = occlusion_map

        return out_dict
    

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, div):
        super().__init__()
        self.q = nn.Conv3d(in_channels, in_channels // div, kernel_size=1)
        self.k = nn.Conv3d(in_channels, in_channels // div, kernel_size=1)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, _, d, h, w = x.size()
        q = self.q(x).view(batch, -1, d * h * w).permute(0, 2, 1)
        k = self.k(x).view(batch, -1, d * h * w)
        v = self.v(x).view(batch, -1, d * h * w)

        attn = torch.bmm(q, k)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v.permute(0, 2, 1))
        out = out.view(batch, -1, d, h, w)

        return self.gamma * out + x

class FlowHead3D(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64):
        super(FlowHead3D, self).__init__()
        self.conv1 = nn.Conv3d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv3d(hidden_dim, 3, 3, padding=1)  # Output 3 channels for 3D flow
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU3D(nn.Module):
    def __init__(self, hidden_dim=32, input_dim=32 + 64):
        super(ConvGRU3D, self).__init__()
        self.convz = nn.Conv3d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv3d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv3d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h

class MotionEncoder3D(nn.Module):
    def __init__(self, args):
        super(MotionEncoder3D, self).__init__()
        cor_planes = args.corr_levels * (2 * args.corr_radius + 1) ** 3  # Corrected for 3D
        self.convc1 = nn.Conv3d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv3d(256, 128, 3, padding=1)
        self.convf1 = nn.Conv3d(3, 128, 7, padding=3)  # Input 3 channels for 3D flow
        self.convf2 = nn.Conv3d(128, 64, 3, padding=1)
        self.conv = nn.Conv3d(64 + 128, 64 - 3, 3, padding=1)  # Adjusted output channels

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class UpdateBlock3D(nn.Module):
    def __init__(self, args, hidden_dim=32):
        super(UpdateBlock3D, self).__init__()
        self.args = args
        self.encoder = MotionEncoder3D(args) # 64 channel out
        self.gru = ConvGRU3D(hidden_dim=hidden_dim, input_dim=hidden_dim + self.encoder.conv.out_channels + 3)
        self.flow_head = FlowHead3D(hidden_dim, hidden_dim=64)
        
        #TODO: Mask in the original RAFT

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1) # 32 + 64

        net = self.gru(net, inp) #32, 64+32
        delta_flow = self.flow_head(net)

        return net, delta_flow


class Args:
    def __init__(self, corr_radius, corr_levels, hidden_dim, context_dim, context_params) -> None:
        self.corr_radius = corr_radius
        self.corr_levels = corr_levels
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim