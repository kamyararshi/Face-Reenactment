import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.util import HourGlass2D

class ExpressionRefiner(nn.Module):
    def __init__(self, 
            block_expansion,
            num_kp,
            num_blocks,
            max_features,
            image_channel,
            feature_channel=None,
            estimate_jacobian=None,
            ) -> None:
        super().__init__()
        self.hourglass = HourGlass2D(
            block_expansion=block_expansion,
            in_features=(num_kp+image_channel),
            num_blocks=num_blocks,
            max_features=max_features,
        )
        self.predictor = nn.Conv2d(
            in_channels=self.hourglass.out_filters,
            out_channels=image_channel,
            kernel_size=5,
            stride=1,
            padding=2
        )
        
        # self.projection = nn.Sequential([
            
        # ])
    
    def forward(self, predicted: torch.Tensor, heatmaps: torch.Tensor):
        # TODO: Try a learnable convolution approach for projecting the heatmaps to image space
        heatmaps_2d = torch.argmax(heatmaps, dim=2).to(torch.float32)
        heatmaps_resized = F.interpolate(heatmaps_2d, size=(256, 256), mode='bilinear', align_corners=False)
        heatmaps_resized = (heatmaps_resized - heatmaps_resized.min()) / (heatmaps_resized.max() - heatmaps_resized.min())
        
        x = torch.cat((heatmaps_resized, predicted), dim=1)
        out = self.hourglass(x)

        # Predict the image
        out = torch.relu(self.predictor(out))

        return out
    