import torch
import torch.nn as nn
import torch.nn.functional as F

#### Adaptive Group Normalization ####
class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_groups, num_features, num_affine_params=2):
        super(AdaptiveGroupNorm, self).__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.num_affine_params = num_affine_params

        # Learnable parameters for affine transformation
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        B, C, D, H, W = x.size()
        x = x.view(B * self.num_groups, -1)

        # Calculate mean and variance
        mean = torch.mean(x, dim=1, keepdim=True)
        var = torch.var(x, dim=1, keepdim=True, unbiased=False)

        # Normalize within each group
        x = (x - mean) / torch.sqrt(var + 1e-5)

        # Reshape to original shape
        x = x.view(B, C, D, H, W)

        # Apply learned affine transformation
        weight = self.weight.view(1, C, 1, 1, 1)
        bias = self.bias.view(1, C, 1, 1, 1)
        x = x * weight + bias

        return x

#### Custom Residual Block with AdaptiveGroupNorm ####
# Creates 3D res blocks by defaault
class CustomResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_dim=3, kernel_size=3, stride=1, padding=1, padding_mode="zeros", downsample = None):
        super(CustomResidualBlock, self).__init__()
        self.stride = stride
        self.down_sample = 1/stride
        self.conv_dim = conv_dim

        # If we want ResidualBlock2D
        if self.conv_dim == 2:
            self.conv1 = nn.Sequential(
                # AdaptiveGroupNorm(1, in_channels),
                nn.BatchNorm2d(in_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding=padding, padding_mode=padding_mode), #TODO: Do we need padding?
            )
            self.conv2 = nn.Sequential(
                # AdaptiveGroupNorm(1, out_channels),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = stride, padding=padding, padding_mode=padding_mode), #TODO: Do we need padding?
            )
            self.skip_connection = \
                nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding=padding, padding_mode=padding_mode) 
            
        # If we want ResidualBlock3D
        elif self.conv_dim == 3:
            self.conv1 = nn.Sequential(
                # AdaptiveGroupNorm(1, in_channels),
                nn.BatchNorm3d(in_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding=padding, padding_mode=padding_mode), #TODO: Do we need padding?
            )
            self.conv2 = nn.Sequential(
                # AdaptiveGroupNorm(1, out_channels),
                nn.BatchNorm3d(out_channels),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size = kernel_size, stride = stride, padding=padding, padding_mode=padding_mode), #TODO: Do we need padding?
            )
            self.skip_connection = \
                nn.Conv3d(in_channels, out_channels, kernel_size = kernel_size, stride = stride, padding=padding, padding_mode=padding_mode) 
        
        else:
            raise ValueError(f"conv_dim should be either 2 or 3 but got {conv_dim}")
        
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x):
        # Check if we have stride=1 in residual block
        if self.stride == 1:
            residual = x
            out = self.conv1(x)
            out = self.conv2(out)
            residual = self.skip_connection(residual)
            out += residual

        # Else we will need downsampling residual (in out+residual)
        # TODO: Replace iterpolate with Conv layer to have learnable parameters
        else:
            residual = x
            out = self.conv1(x)
            out = self.conv2(out)
            residual = self.skip_connection(residual)
            # Downsampling residual
            residual = F.interpolate(residual, scale_factor=self.down_sample, mode='trilinear', align_corners=True)
            out += residual

        return out
    

#### Expression Warping ####
class ExpressionWarper(torch.nn.Module):
    def __init__(self, in_channels: int, estimate_occlusion: bool=True, **kwargs) -> None:
        super(ExpressionWarper, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, 512, 1)
        
        self.res1 = CustomResidualBlock(512, 256, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.up1 = nn.Upsample(scale_factor=(2, 2, 2))
        
        self.res2 = CustomResidualBlock(256, 128, kernel_size=3, padding=1)
        self.up2 = nn.Upsample(scale_factor=(2, 2, 2))
        
        self.res3 = CustomResidualBlock(128, 64, kernel_size=3, padding=1)
        self.up3 = nn.Upsample(scale_factor=(1, 2, 2))
        
        self.res4 = CustomResidualBlock(64, 32, kernel_size=3, padding=1)
        self.up4 = nn.Upsample(scale_factor=(1, 2, 2))
        
        self.gn = nn.BatchNorm3d(32)
        self.relu = nn.ReLU()
        
        # Adjust conv2 to produce the desired output shape
        self.conv2 = nn.Conv3d(32, 3, kernel_size=(3, 4, 4), padding=(1, 1, 1), padding_mode="replicate")
        
        # Final upsampling to reach (16, 64, 64)
        self.up5 = nn.Upsample(size=(16, 64, 64), mode="trilinear", align_corners=False)
        
        if estimate_occlusion:
            self.occlusion = nn.Conv2d(self.conv2.out_channels*16, 1, kernel_size=7, padding=3)
        else:
            self.occlusion = None

    def forward(self, x):
        x = x.unsqueeze(-1)  # Shape: (bs, 128, 1)
        x = self.conv1(x)  # Shape: (bs, 512, 1)
        x = x.unsqueeze(2).unsqueeze(3)  # Shape: (bs, 512, 1, 1, 1)
        
        x = self.res1(x)
        x = self.up1(x)
        x = self.res2(x)
        x = self.up2(x)
        
        x = self.res3(x)
        x = self.up3(x)
        x = self.res4(x)
        x = self.up4(x)
        
        x = self.gn(x)
        x = self.relu(x)
        x = self.conv2(x)  # Shape: (bs, 3, 16, 32, 32)
        x = self.up5(x)  # Shape: (bs, 3, 16, 64, 64)
        
        out_dict = {}
        w_em = F.leaky_relu(x)
        out_dict['w_em'] = w_em

        if self.occlusion:

            bs, c, d, h, w = w_em.shape
            w_em = w_em.view(bs, -1, h, w)
            occlusion_map = torch.sigmoid(self.occlusion(w_em))
            out_dict['occlusion_map'] = occlusion_map

        return out_dict
