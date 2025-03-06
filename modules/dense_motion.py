from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, SPADEResnetBlock3D, make_coordinate_grid, kp2gaussian

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

        # self.compress = nn.Conv3d(feature_channel, compress, kernel_size=1)
        # self.norm = BatchNorm3d(compress, affine=True)

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

        out_dict = dict()
        sparse_motion = self.create_sparse_motions(feature, kp_driving, kp_source)
        deformed_feature = self.create_deformed_feature(feature, sparse_motion) #NOTE: Warps the feature with the sparse_motion

        heatmap = self.create_heatmap_representations(deformed_feature, kp_driving, kp_source)

        # heatmap shape (bs, num_kp+1, 1, d, h, w)
        sparse_motion = sparse_motion.permute(0, 1, 5, 2, 3, 4)    # (bs, num_kp+1, 3, d, h, w)
        deformation = (sparse_motion * heatmap).sum(dim=1)            # (bs, 3, d, h, w)
        deformation = deformation.permute(0, 2, 3, 4, 1)           # (bs, d, h, w, 3)

        out_dict['deformation'] = deformation

        if self.occlusion:
            bs, c, d, h, w = prediction.shape
            prediction = prediction.view(bs, -1, h, w)
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map

        return out_dict
    

#TODO: Replace this with our actual SPADE layer.
class SPADE(nn.Module):
    def __init__(self, in_channels, norm_nc, ks=3):
        super().__init__()
        self.norm = nn.BatchNorm3d(norm_nc, affine=False)
        self.conv = nn.Conv3d(in_channels, 2*norm_nc, kernel_size=ks, padding=ks//2)

    def forward(self, x, segmap):
        normalized = self.norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.conv(segmap)
        gamma, beta = torch.chunk(actv, 2, dim=1)
        out = normalized * (1 + gamma) + beta
        return out

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.q = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.k = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
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

class MotionRefinementModule(nn.Module):
    def __init__(self, feature_channels, corr_channels, motion_channels, label_nc, attention_channels):
        super().__init__()
        self.conv_corr = nn.Conv3d(corr_channels, attention_channels, kernel_size=3, padding=1)
        self.attention = AttentionBlock(attention_channels)
        self.conv_motion = nn.Conv3d(motion_channels, 32, kernel_size=3, padding=1)
        self.spade_resnet = SPADEResnetBlock3D(feature_channels, 64, 'none', label_nc)
        self.conv_residual = nn.Sequential(
            nn.Conv3d(64 + 64 + 32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.2),
            nn.Conv3d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.attention_channels = attention_channels

    def forward(self, feature, corr, motion, segmap):
        feature = self.spade_resnet(feature, segmap)
        corr = self.attention(self.conv_corr(corr))
        motion = self.conv_motion(motion)
        combined = torch.cat([feature, corr, motion], dim=1)
        residual = self.conv_residual(combined)
        return residual

class MultiScaleMotionRefinement(nn.Module):
    def __init__(self, spade_channels):
        super().__init__()
        self.refinement_modules = nn.ModuleList([
            MotionRefinementModule(32, 32*32*1, 3, spade_channels, 64),
            MotionRefinementModule(64, 16*16*1, 3, spade_channels, 64),
            MotionRefinementModule(128, 8*16*1, 3, spade_channels, 64),
            MotionRefinementModule(256, 4*16*1, 3, spade_channels, 64)
        ])

    def forward(self, features, corrs, motions, segmaps, scale: int):
        return self.refinement_modules[scale](features, corrs, motions, segmaps)