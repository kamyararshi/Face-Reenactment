import torch
import torch.nn as nn 
import numpy as np
import copy
from .util_loss import resnet18

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def pos_embed(self, src, pos):
        batch_pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
        return src + batch_pos
        

    def forward(self, src, pos):
                # src_mask: Optional[Tensor] = None,
                # src_key_padding_mask: Optional[Tensor] = None):
                # pos: Optional[Tensor] = None):

        q = k = self.pos_embed(src, pos)
        #print("Q:", q.shape, "K:", k.shape, "V:", src.shape)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class Model(nn.Module):
    """
    GazeTR Model trained on image_shape=(220, 220) from gaze360 dataset
    """
    def __init__(self, criterion: str):
        super(Model, self).__init__()
        maps = 32
        nhead = 8
        dim_feature = 7*7
        dim_feedforward=512
        dropout = 0.1
        num_layers=6

        self.base_model = resnet18(pretrained=False, maps=maps)

        # d_model: dim of Q, K, V 
        # nhead: seq num
        # dim_feedforward: dim of hidden linear layers
        # dropout: prob

        encoder_layer = TransformerEncoderLayer(
                  maps, 
                  nhead, 
                  dim_feedforward, 
                  dropout)

        encoder_norm = nn.LayerNorm(maps) 
        # num_encoder_layer: deeps of layers 

        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self.cls_token = nn.Parameter(torch.randn(1, 1, maps))

        self.pos_embedding = nn.Embedding(dim_feature+1, maps)
        self.feed = nn.Linear(maps, 3) # Changed to 3 instead of 2 for Gaze360 dataset
            
        if criterion == 'l1':
            self.criterion = nn.L1Loss()
        elif criterion == 'l2':
            self.criterion = nn.MSELoss()
        elif criterion == 'cosine':
            self.criterion = nn.CosineSimilarity()
        elif criterion == 'l1_smooth':
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown criterion: {criterion}. Must be from l1 or l2 or cosine")


    def forward(self, x_in: torch.tensor):
        feature = self.base_model(x_in)
        batch_size = feature.size(0)
        feature = feature.flatten(2)
        feature = feature.permute(2, 0, 1)
        
        cls = self.cls_token.repeat( (1, batch_size, 1))
        feature = torch.cat([cls, feature], 0)
        feature_size = feature.shape[0]
        
        #position = torch.from_numpy(np.arange(0, 50)).cuda()
        position = torch.from_numpy(np.arange(0, feature_size)).to(x_in.device)

        pos_feature = self.pos_embedding(position)

        # feature is [HW, batch, channel]
        feature = self.encoder(feature, pos_feature)
  
        feature = feature.permute(1, 2, 0)

        feature = feature[:,:,0]

        gaze = self.feed(feature)
        
        return gaze
    
    @torch.no_grad() # impacts the autograd engine and deactivate it. It will reduce memory usage and speed up computations
    def inference(self, x_in: torch.tensor):
        feature = self.base_model(x_in)
        batch_size = feature.size(0)
        feature = feature.flatten(2)
        feature = feature.permute(2, 0, 1)
        
        cls = self.cls_token.repeat( (1, batch_size, 1))
        feature = torch.cat([cls, feature], 0)
        
        position = torch.arange(0, 50).to(x_in.device)

        pos_feature = self.pos_embedding(position)

        # feature is [HW, batch, channel]
        feature = self.encoder(feature, pos_feature)
  
        feature = feature.permute(1, 2, 0)

        feature = feature[:,:,0]

        gaze = self.feed(feature)
        
        return gaze

    def loss(self, x_in, x_true):
        x_in = nn.functional.interpolate(x_in, (220, 220), mode='bicubic', align_corners=False)
        x_true = nn.functional.interpolate(x_true, (220, 220), mode='bicubic', align_corners=False)
        gaze = self.inference(x_in)
        gaze_true = self.inference(x_true)
        loss = self.criterion(gaze, gaze_true) 
        return loss
