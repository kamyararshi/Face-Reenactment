"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo

import torchvision.models.vgg as weights19
from torchvision import models

from SegFace.network import get_model
import cv2


# Credit: https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/losses.py
def TernaryLoss(im, im_warp, max_distance=1):
    patch_size = 2 * max_distance + 1

    def _rgb_to_grayscale(image):
        grayscale = (
            image[:, 0, :, :] * 0.2989
            + image[:, 1, :, :] * 0.5870
            + image[:, 2, :, :] * 0.1140
        )
        return grayscale.unsqueeze(1)

    def _ternary_transform(image):
        intensities = _rgb_to_grayscale(image) * 255
        out_channels = patch_size * patch_size
        w = torch.eye(out_channels).view((out_channels, 1, patch_size, patch_size))
        weights = w.type_as(im)
        patches = F.conv2d(intensities, weights, padding=max_distance)
        transf = patches - intensities
        transf_norm = transf / torch.sqrt(0.81 + torch.pow(transf, 2))
        return transf_norm

    def _hamming_distance(t1, t2):
        dist = torch.pow(t1 - t2, 2)
        dist_norm = dist / (0.1 + dist)
        dist_mean = torch.mean(dist_norm, 1, keepdim=True)  # instead of sum
        return dist_mean

    def _valid_mask(t, padding):
        n, _, h, w = t.size()
        inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
        mask = F.pad(inner, [padding] * 4)
        return mask

    t1 = _ternary_transform(im)
    t2 = _ternary_transform(im_warp)
    dist = _hamming_distance(t1, t2)
    mask = _valid_mask(im, max_distance)

    return dist * mask


def SSIM(x, y, md=1):
    patch_size = 2 * md + 1
    C1 = 0.01**2
    C2 = 0.03**2

    mu_x = nn.AvgPool2d(patch_size, 1, 0)(x)
    mu_y = nn.AvgPool2d(patch_size, 1, 0)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(patch_size, 1, 0)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(patch_size, 1, 0)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d
    dist = torch.clamp((1 - SSIM) / 2, 0, 1)
    return dist

def gradient(data):
    D_dy = data[..., 1:, :] - data[..., :-1, :]
    D_dx = data[..., :, 1:] - data[..., :, :-1]
    return D_dx, D_dy

def gradient_flow(data):
    D_dy = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]
    D_dx = data[:, :, :, 1:, :] - data[:, :, :, :-1, :]
    D_dz = data[:, 1:, :, :, :] - data[:, :-1, :, :, :]
    return D_dx, D_dy, D_dz

def get_image_edge_weights(image, alpha=10):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)
    return weights_x, weights_y


def get_full_seg_edge_weights(full_seg):
    weights_y = (full_seg[..., 1:, :] - full_seg[..., :-1, :] == 0).float()
    weights_x = (full_seg[..., :, 1:] - full_seg[..., :, :-1] == 0).float()
    return weights_x, weights_y


def smooth_grad_1st(flo, image, reduce_image=False, edge="image", **kwargs):
    B, _, H, W, _ = flo.shape
    if reduce_image:
        image = F.interpolate(image, (H,W), mode="bilinear", align_corners=False)
    
    if edge == "image":
        weights_x, weights_y = get_image_edge_weights(image, kwargs["alpha"])
    elif edge == "full_seg":
        weights_x, weights_y = get_full_seg_edge_weights(kwargs["full_seg"])

    if not reduce_image:
        weights_x = F.interpolate(weights_x, (H,W-1), mode="bilinear", align_corners=False)
        weights_y = F.interpolate(weights_y, (H-1,W), mode="bilinear", align_corners=False)
    #[B, 16, 32, 32, 3]
    dx, dy, dz = gradient_flow(flo)
    loss_x = weights_x.unsqueeze(-1) * dx.abs()
    loss_y = weights_y.unsqueeze(-1) * dy.abs()
    loss_z = dz.abs()

    return loss_x.mean() / 3.0 + loss_y.mean() / 3.0 + loss_z.mean() / 3.0


def smooth_grad_2nd(flo, image, edge="image", **kwargs):
    if edge == "image":
        weights_x, weights_y = get_image_edge_weights(image, kwargs["alpha"])
    elif edge == "full_seg":
        weights_x, weights_y = get_full_seg_edge_weights(kwargs["full_seg"])

    dx, dy = gradient_flow(flo)
    dx2, dxdy = gradient_flow(dx)
    dydx, dy2 = gradient_flow(dy)

    loss_x = weights_x[:, :, :, 1:] * dx2.abs()
    loss_y = weights_y[:, :, 1:, :] * dy2.abs()
    # loss_x = weights_x[:, :, :, 1:] * (torch.exp(dx2.abs() * 100) - 1) / 100.
    # loss_y = weights_y[:, :, 1:, :] * (torch.exp(dy2.abs() * 100) - 1) / 100.

    return loss_x.mean() / 2.0 + loss_y.mean() / 2.0



# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False, weights=weights19.VGG19_Weights.IMAGENET1K_V1):
        super().__init__()
        vgg_pretrained_features = models.vgg19(weights=weights).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
    
# VGG16 architecter, used for the perceptual loss using the pretrained VGGFace network
class VGG16(nn.Module):
    def __init__(self, requires_grad=False, weights='DEFAULT'):
        super().__init__()
        self.vgg16_pretrained_features = models.vgg16(weights=weights).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), self.vgg16_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), self.vgg16_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), self.vgg16_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), self.vgg16_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), self.vgg16_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    
        if not requires_grad:
            for param in self.parameters(): 
                param.requires_grad=False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = h_relu5
        return out


####### Gaze Estimation Backbone Network #######
# ResNet18 architecture for Gaze Estimation backbone (from gazeTR https://github.com/yihuacheng/GazeTR?tab=readme-ov-file)

model_urls = {
     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth' 
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, maps=32):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        self.conv = nn.Sequential(
            nn.Conv2d(512, maps, 1),
            nn.BatchNorm2d(maps),
            nn.ReLU(inplace=True)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)      
        x = self.conv(x)
        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']),strict=False)
    return model

class FaceParser:
    def __init__(self, device: str,
                 input_resolution: int=256,
                 backbone: str='segface_celeb',
                 model_name: str='swin_base',
                 weight_path: str="./SegFace/weights/swinb_celeba_256/model_299.pt"):
        self.device = device
        self.input_resolution = input_resolution
        self.backbone = backbone
        self.model_name = model_name
        self.weight_path = weight_path
        self.model = self.get_faceparser(device, input_resolution, backbone, model_name, weight_path)

    @torch.no_grad()
    def __call__(self, images, keep:list, *args, **kwds):
        """Zeros out other pixels except for the masked ones and returns either:
            1. Images with other pixels zeroed out
            2. A boolean mask of the same shape as the images where masked pixels are true and others false

        Args:
            images (torch.Tensor): Images of shape (B, C, H, W)
            masks (torch.Tensor): Segmentation masks of shape (B, H, W)
            keep (list): List of labels to keep
                1. 'eyes'
                2. 'lips'
                3. 'ears'
                4. 'eyebrows'
                5. 'mouth'
                6. 'nose'
                7. 'hair'
                8. 'neck'
                9. 'hat'
                10. 'skin'
                11. 'cloth'
                12. 'background'
        Returns:
            masked_images (torch.Tensor): Images with other pixels zeroed out
        """
        masks = self.get_mask(images, input_resolution=256)
        masked_images = self.seperate_masks(images, masks, keep=keep, mode='image')
        return masked_images

    def get_faceparser(self, device: str,
                        input_resolution: int=256,
                        backbone: str='segface_celeb',
                        model_name: str='swin_base',
                        weight_path: str="./weights/swinb_celeba_256/model_299.pt"):

        model = get_model(backbone, input_resolution, model_name).to(device)
        model.eval()
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint['state_dict_backbone'])
        return model

    def seperate_masks(self, images: torch.Tensor,
                       masks: torch.Tensor,
                       keep: list,
                       mode: str='loss') -> torch.Tensor:
        """Zeros out other pixels except for the masked ones
        Args:
            images (torch.Tensor): Images of shape (B, C, H, W)
            masks (torch.Tensor): Segmentation masks of shape (B, H, W)
            keep (list): List of labels to keep
        1. 'eyes'
        2. 'lips'
        3. 'ears'
        4. 'eyebrows'
        5. 'mouth'
        6. 'nose'
        7. 'hair'
        8. 'neck'
        9. 'hat'
        10. 'skin'
        11. 'cloth'
        12. 'background'
        Returns:
            masked_images (torch.Tensor): Images with other pixels zeroed out
        """
        assert mode in ['image', 'loss'], "mode should be either 'image' or 'loss'"
        
        names = {'eyes': (8, 9), 'lips': (12, 13), 'ears': (4, 5), 'eyebrows': (6, 7), 'mouth': 11, 'nose': 10,
                'hair': 14, 'neck': 1, 'hat': 16, 'skin': 2, 'cloth': 3, 'background': 0,'eye_g': 15, 'ear_r': 17, 'neck_l': 18}
        
        combined_mask = torch.zeros_like(masks, dtype=torch.bool, device=masks.device)
        for key in keep:
            label = names[key]
            if isinstance(label, tuple):
                combined_mask |= (masks == label[0]) | (masks == label[1])
            else:
                combined_mask |= (masks == label)

        if mode == 'image':
            masked_images = torch.zeros_like(images, device=images.device)
            masked_images[combined_mask.unsqueeze(1).expand_as(images)] = images[combined_mask.unsqueeze(1).expand_as(images)]
        elif mode == 'loss':
            masked_images = torch.zeros_like(images, device=images.device, dtype=torch.bool)
            masked_images[combined_mask.unsqueeze(1).expand_as(images)] = 1
        return masked_images
        
        
    @torch.no_grad()
    def get_mask(self,
                 images: torch.Tensor,
                input_resolution: int=256):
        """ Returns face masks using face parsing"""
        seg_output = self.model(images, labels=None, dataset=None)
        if seg_output.shape[-2:] != (input_resolution, input_resolution):
            mask = F.interpolate(seg_output, size=(input_resolution, input_resolution), mode='bilinear', align_corners=False)
        else:
            mask = seg_output
        mask = mask.softmax(dim=1)
        preds = torch.argmax(mask, dim=1)
        return preds