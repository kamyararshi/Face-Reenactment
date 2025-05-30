from torch import nn

import torch.nn.functional as F
import torch

from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d

import torch.nn.utils.spectral_norm as spectral_norm
import re


def coords_grid(batch, d, ht, wd, device):
    coords = torch.meshgrid(
        torch.arange(d, device=device),
        torch.arange(ht, device=device),
        torch.arange(wd, device=device),
                            )
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1, 1)

def trilinear_sampler(img, coords, mask=False):
    """
    Trilinear sampling for 3D correlation volumes
    """
    b, c, dd, hh, ww = img.shape
    w_coords, h_coords, d_coords = coords[..., 0], coords[..., 1], coords[..., 2]
    
    w_coords = 2 * w_coords / (ww - 1) - 1
    h_coords = 2 * h_coords / (hh - 1) - 1
    d_coords = 2 * d_coords / (dd - 1) - 1
    
    grid = torch.stack([w_coords, h_coords, d_coords], dim=-1).to(img.dtype)
    img = F.grid_sample(img, grid, align_corners=True, mode='bilinear') 

    if mask:
        mask = (d_coords > -1) & (h_coords > -1) & (w_coords > -1) & (d_coords < 1) & (h_coords < 1) & (w_coords < 1)
        return img, mask.float()
    
    return img

def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    mean = kp['value']

    coordinate_grid = make_coordinate_grid(spatial_size, mean.type(), device=mean.device)
    number_of_leading_dimensions = len(mean.shape) - 1
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape
    coordinate_grid = coordinate_grid.view(*shape)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 3)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean)

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    return out

def make_coordinate_grid_2d(spatial_size, type, device):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w, device=device).type(type)
    y = torch.arange(h, device=device).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed.to(device)

# Extend to 3D TPS grid if TPS is used
def make_coordinate_grid_3d(spatial_size, type, device):
    """
    Create a 3D coordinate grid for the given spatial size.
    """
    d, h, w = spatial_size
    x = torch.linspace(-1, 1, steps=w, device=device).type(type)
    y = torch.linspace(-1, 1, steps=h, device=device).type(type)
    z = torch.linspace(-1, 1, steps=d, device=device).type(type)
    # grid = torch.meshgrid(x, y, z, indexing='ij')
    grid = torch.meshgrid(x, y, z)
    grid = torch.stack(grid, dim=-1)
    return (grid.view(-1, 3)).to(device)

def make_coordinate_grid(spatial_size, type, device):
    d, h, w = spatial_size
    x = torch.arange(w, device=device).type(type)
    y = torch.arange(h, device=device).type(type)
    z = torch.arange(d, device=device).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    z = (2 * (z / (d - 1)) - 1)
   
    yy = y.view(1, -1, 1).repeat(d, 1, w)
    xx = x.view(1, 1, -1).repeat(d, h, 1)
    zz = z.view(-1, 1, 1).repeat(1, h, w)

    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3), zz.unsqueeze_(3)], 3)

    return meshed

def mesh_grid(bs, d, h, w):
    x = torch.arange(w).view(1, 1, 1, 1, w).expand(bs, 1, d, h, w)
    y = torch.arange(h).view(1, 1, 1, h, 1).expand(bs, 1, d, h, w)
    z = torch.arange(d).view(1, 1, d, 1, 1).expand(bs, 1, d, h, w)
    base_grid = torch.cat((x, y, z), 1).float()
    return base_grid

def norm_grid(grid):
    _, _, d, h, w = grid.shape
    grid[:, 0, :, :, :] = grid[:, 0, :, :, :] / (w - 1) * 2 - 1
    grid[:, 1, :, :, :] = grid[:, 1, :, :, :] / (h - 1) * 2 - 1
    grid[:, 2, :, :, :] = grid[:, 2, :, :, :] / (d - 1) * 2 - 1
    return grid.permute(0, 2, 3, 4, 1)

class ResBottleneck(nn.Module):
    def __init__(self, in_features, stride):
        super(ResBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features//4, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=in_features//4, out_channels=in_features//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(in_channels=in_features//4, out_channels=in_features, kernel_size=1)
        self.norm1 = BatchNorm2d(in_features//4, affine=True)
        self.norm2 = BatchNorm2d(in_features//4, affine=True)
        self.norm3 = BatchNorm2d(in_features, affine=True)

        self.stride = stride
        if self.stride != 1:
            self.skip = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=1, stride=stride)
            self.norm4 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        if self.stride != 1:
            x = self.skip(x)
            x = self.norm4(x)
        out += x
        out = F.relu(out)
        return out


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class ResBlock3d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm3d(in_features, affine=True)
        self.norm2 = BatchNorm3d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out

class UpBlock3d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock3d, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        # out = F.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear')
        out = F.interpolate(x, scale_factor=(1, 2, 2)) #TODO: Use convTranspose to have learnable params
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class DownBlock3d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock3d, self).__init__()
        '''
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups, stride=(1, 2, 2))
        '''
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm3d(out_features, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1, lrelu=False):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        if lrelu:
            self.ac = nn.LeakyReLU()
        else:
            self.ac = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.ac(out)
        return out


class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock3d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock3d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        # self.out_filters = block_expansion
        self.out_filters = block_expansion + in_features

        self.conv = nn.Conv3d(in_channels=self.out_filters, out_channels=self.out_filters, kernel_size=3, padding=1)
        self.norm = BatchNorm3d(self.out_filters, affine=True)

    def forward(self, x):
        out = x.pop()
        # for up_block in self.up_blocks[:-1]:
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        # out = self.up_blocks[-1](out)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out


class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Encoder2D(nn.Module):
    """
    Hourglass Encoder for 2D data
    """
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder2D, self).__init__()
        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(
                in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                min(max_features, block_expansion * (2 ** (i + 1))),
                kernel_size=3, padding=1
            ))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs

class Decoder2D(nn.Module):
    """
    Hourglass Decoder for 2D data
    """
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder2D, self).__init__()
        up_blocks = []
        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features
        self.conv = nn.Conv2d(in_channels=self.out_filters, out_channels=self.out_filters, kernel_size=3, padding=1)
        self.norm = BatchNorm2d(self.out_filters, affine=True)

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out

# HourGlass2D Module
class HourGlass2D(nn.Module):
    """
    Hourglass architecture for 2D data
    """
    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(HourGlass2D, self).__init__()
        self.encoder = Encoder2D(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder2D(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AttentionKPHourglass(nn.Module):
    """
    Hourglass architecture.
    """ 

    def __init__(self, block_expansion, in_features, reshape_features, reshape_depth, num_blocks=3, max_features=256):
        super(AttentionKPHourglass, self).__init__()
        
        self.down_blocks = nn.Sequential()
        self.attention = nn.Sequential()
        for i in range(num_blocks):
            #TODO: Maybe only one layer attention at the end or only at the beginning?
            # self.attention.add_module('attention'+ str(i), AttentionBlock2D(in_features if i == 0 else min(max_features, block_expansion * (2 ** i))))
            self.down_blocks.add_module('down'+ str(i), DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                                                   min(max_features, block_expansion * (2 ** (i + 1))),
                                                                   kernel_size=3, padding=1))

        in_filters = min(max_features, block_expansion * (2 ** num_blocks))
        self.conv = nn.Conv2d(in_channels=in_filters, out_channels=reshape_features, kernel_size=1)

        self.up_blocks = nn.Sequential()
        for i in range(num_blocks):
            in_filters = min(max_features, block_expansion * (2 ** (num_blocks - i)))
            out_filters = min(max_features, block_expansion * (2 ** (num_blocks - i - 1)))
            self.up_blocks.add_module('up'+ str(i), UpBlock3d(in_filters, out_filters, kernel_size=3, padding=1))

        self.reshape_depth = reshape_depth
        self.out_filters = out_filters

    def forward(self, x):
        for i in range(len(self.down_blocks)):
            # x = self.attention[i](x)
            x = self.down_blocks[i](x)
        
        out = self.conv(x)
        bs, c, h, w = out.shape
        out = out.view(bs, c//self.reshape_depth, self.reshape_depth, h, w)
        out = self.up_blocks(out)

        return out
    

class ContextHourglass(nn.Module):
    """
    Hourglass architecture.
    """ 

    def __init__(self, block_expansion, image_channel, scale_factor, reshape_channel, reshape_depth, out_features, num_blocks=3, max_features=256):
        super(ContextHourglass, self).__init__()
        
        if scale_factor != 1:
            self.down = AntiAliasInterpolation2d(image_channel, scale_factor)
            in_features = image_channel

        self.down_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.down_blocks.add_module('down'+ str(i), DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                                                   min(max_features, block_expansion * (2 ** (i + 1))),
                                                                   kernel_size=3, padding=1))

        in_filters = min(max_features, block_expansion * (2 ** num_blocks))
        self.conv = nn.Conv2d(in_channels=in_filters, out_channels=reshape_channel, kernel_size=1)

        self.up_blocks = nn.Sequential()
        for i in range(num_blocks):
            in_filters = min(max_features, block_expansion * (2 ** (num_blocks - i)))
            out_filters = out_features if i==num_blocks-1 else min(max_features, block_expansion * (2 ** (num_blocks - i - 1)))
            self.up_blocks.add_module('up'+ str(i), UpBlock3d(in_filters, out_filters, kernel_size=3, padding=1))

        self.reshape_depth = reshape_depth
        self.out_filters = out_filters

    def forward(self, x):
        x = self.down(x)
        out = self.down_blocks(x)
        out = self.conv(out)
        bs, c, h, w = out.shape
        out = out.view(bs, c//self.reshape_depth, self.reshape_depth, h, w)
        out = self.up_blocks(out)

        return out
        

class AttentionBlock2D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_qk = in_channels // 8 if in_channels // 8 > 0 else 1
        self.q = nn.Conv2d(in_channels, out_qk, kernel_size=1)
        self.k = nn.Conv2d(in_channels, out_qk, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, c, h, w = x.size()
        q = self.q(x).view(batch, -1, h * w).permute(0, 2, 1)
        k = self.k(x).view(batch, -1, h * w)
        v = self.v(x).view(batch, -1, h * w)

        attn = torch.bmm(q, k)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v.permute(0, 2, 1))
        out = out.view(batch, -1, h, w)

        return self.gamma * out + x
    

class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 128

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU())
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out
    

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, norm_G, label_nc, use_se=False, dilation=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.use_se = use_se
        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        # apply spectral norm if specified
        if 'spectral' in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        # define normalization layers
        self.norm_0 = SPADE(fin, label_nc)
        self.norm_1 = SPADE(fmiddle, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)

    def forward(self, x, seg1):
        x_s = self.shortcut(x, seg1)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg1)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg1)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg1):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg1))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class SPADE3D(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super().__init__()

        self.param_free_norm = nn.BatchNorm3d(norm_nc, affine=False)
        nhidden = 128

        self.mlp_shared = nn.Sequential(
            nn.Conv3d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU())
        self.mlp_gamma = nn.Conv3d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv3d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out

class SPADEResnetBlock3D(nn.Module):
    def __init__(self, fin, fout, norm_G, label_nc, use_se=False, dilation=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.use_se = use_se
        # create conv layers
        self.conv_0 = nn.Conv3d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1 = nn.Conv3d(fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation)
        if self.learned_shortcut:
            self.conv_s = nn.Conv3d(fin, fout, kernel_size=1, bias=False)
        # apply spectral norm if specified
        if 'spectral' in norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        # define normalization layers
        self.norm_0 = SPADE3D(fin, label_nc)
        self.norm_1 = SPADE3D(fmiddle, label_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE3D(fin, label_nc)

    def forward(self, x, seg1):
        x_s = self.shortcut(x, seg1)
        dx = self.conv_0(self.actvn(self.norm_0(x, seg1)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg1)))
        out = x_s + dx
        return out

    def shortcut(self, x, seg1):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg1))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)