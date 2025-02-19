import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from modules.dynamic_conv import Dynamic_conv2d
import math


def conv3x3_3d(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, dropout=0.5):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.drop_out = nn.Dropout2d(dropout)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class BasicBlock3d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock3d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3_3d(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.drop_out = nn.Dropout3d(0.5)
        self.conv2 = conv3x3_3d(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += self.drop_out(identity)
        out = self.relu(out)

        return out


class ExpressionEncoder(nn.Module):
    def __init__(self, in_channel=512, d_model=512, encode=False):
        super().__init__()
        self.d_model = d_model
        self.encode = encode
        self.sqrt_d = math.sqrt(self.d_model)
        self.rearange = Rearrange('b c d h w -> b (c d) h w')
        self.block = BasicBlock(inplanes=in_channel, planes=d_model) if self.encode else None
        self.norm = nn.BatchNorm2d(d_model)
        self.convq = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        # self.convkv = Dynamic_conv2d(in_channels=d_model, out_planes=d_model*2, kernel_size=3, ratio=0.25, padding=1,groups=1)
        self.convkv = nn.Conv2d(in_channels=d_model, out_channels=d_model*2, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.rearange(x)
        b,c,w,h = x.shape
        query = F.relu(self.convq(x))
        key,value = F.relu(self.convkv(x)).chunk(2, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b c x y -> b (x y) c'), (query,key,value))
        sim = torch.einsum('b i d, b j d -> b i j', q, k)
        A = torch.softmax(sim/self.sqrt_d,-1)
        residual = torch.einsum('b i j, b j d -> b i d', A, v).contiguous()

        # Add and Norm
        residual = rearrange(residual, 'b (x y) c -> b c x y', x=w,y=h)
        residual = self.norm(residual+x)
        if self.encode:
            residual = self.block(residual)
        return residual
    
class ExpressionEncoder3d(nn.Module):
    def __init__(self, in_channel=512, d_model=512):
        #TODO: Does not work yet
        super().__init__()
        self.d_model = d_model
        self.sqrt_d = math.sqrt(self.d_model)
        self.block = BasicBlock3d(inplanes=in_channel, planes=d_model, stride=2)
        self.convq = nn.Conv3d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.convkv = nn.Conv3d(in_channels=d_model, out_channels=d_model*2, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.block(x)
        b,c,w,h = x.shape
        query = F.relu(self.convq(x))
        key,value = F.relu(self.convkv(x)).chunk(2, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b c x y -> b (x y) c'), (query,key,value))
        sim = torch.einsum('b i d, b j d -> b i j', q, k)
        A = torch.softmax(sim/self.sqrt_d,-1)
        residual = torch.einsum('b i j, b j d -> b i d', A, v)
        residual = rearrange(residual, 'b (x y) c -> b c x y', x=w,y=h)
        return residual        


class CrossConvAttention(nn.Module):
    def __init__(self, projection_dim=512, in_channel=512, d_model=512, encode_src=True, encode_drv=False):
        super().__init__()
        self.projection_dim = projection_dim
        self.sqrt_d = math.sqrt(self.projection_dim)
        self.channel_32 = 512
        self.channel_64 = 256
        self.channel_128 = 128
        self.expression_encoder_src = ExpressionEncoder(in_channel=in_channel, d_model=d_model, encode=encode_src)
        self.expression_encoder_drv = ExpressionEncoder(in_channel=in_channel, d_model=d_model, encode=encode_drv)
        self.conv_query = nn.Conv2d(in_channels=d_model, out_channels=self.projection_dim, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels=d_model, out_channels=self.projection_dim, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels=d_model, out_channels=self.projection_dim, kernel_size=1)
        self.diff_conv = nn.Conv3d(in_channels=32, out_channels=32, kernel_size= 1, padding=0)
        self.diff_conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size= 1, padding=0)

    def forward(self, features_deformed, features_driving):
        # Encode warped source features and driving features for expression
        b, x, y, i, j = features_deformed.shape
        out = {}
        residual_source = self.expression_encoder_src(features_deformed)
        out['expr_source'] = residual_source
        b, c, w, h = residual_source.shape
        residual_driving = self.expression_encoder_drv(features_driving)
        out['expr_driving'] = residual_driving
        # Compute Cross-Attention
        # Compute query
        query = self.conv_query(residual_source)
        # Compute key and value
        key = self.conv_key(residual_driving)
        value = self.conv_value(residual_driving)
        q, k, v = map(lambda t: rearrange(t, 'b c x y -> b (x y) c'), (query,key,value))
        sim = torch.einsum('b i d, b j d -> b i j', q, k)
        A = torch.softmax(sim/self.sqrt_d,-1)
        residual = torch.einsum('b i j, b j d -> b i d', A, v)
        residual = rearrange(residual, 'b (x y) c -> b c x y', x=w,y=h)
        out['residual'] = residual

        # Compute difference
        # residual = residual.view(b, x, y, i, j)
        # diff =F.relu( self.diff_conv(features_deformed - residual))
        # out['features_refined'] = F.relu(features_deformed + diff)
        out['features_refined'] = F.relu(self.diff_conv(residual))
        return out