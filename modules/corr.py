import torch
import torch.nn.functional as F
from modules.util import trilinear_sampler, bilinear_sampler


class CorrBlock3D:
    def __init__(self, fmap1: torch.tensor, fmap2: torch.tensor,
                 num_levels: int=4, radius: int=4, single_scale=False):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock3D.corr(fmap1, fmap2)

        # Reshape for pyramid creation
        batch, dd, ht, wd, dim, _, _, _ = corr.shape # Corrected shape
        corr = corr.reshape(batch*dd*ht*wd, 1, dd, ht, wd) # corrected reshape
        self.corr_pyramid.append(corr)
        if not single_scale:
            for i in range(self.num_levels-1):
                corr = F.avg_pool3d(corr, 2, stride=2)
                self.corr_pyramid.append(corr)

    def __call__(self, coords: torch.tensor):
        r = self.radius
        batch, c, d1, h1, w1 = coords.shape
        assert c==3, f"Motion channel should be 3 instead is {c}"

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            
            # Create 3D sampling grid
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dz = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dz, dy, dx), axis=-1)

            # Scale coordinates for current level
            centroid_lvl = coords.permute(0,2,3,4,1).reshape(batch*d1*h1*w1, 1, 1, 1, 3) / 2**i # corrected coordinate order and permute
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2*r+1, 3)
            coords_lvl = centroid_lvl + delta_lvl

            # Sample correlation volume
            corr = trilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, d1, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 4, 1, 2, 3).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, dd, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, dd*ht*wd)
        fmap2 = fmap2.view(batch, dim, dd*ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, dd, ht, wd, 1, dd, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())