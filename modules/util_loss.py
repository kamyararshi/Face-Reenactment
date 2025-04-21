"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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