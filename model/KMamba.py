from typing import Sequence, Tuple, Type, Union

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from model.SwinUNETR import SwinUNETR
from model.Unet import UNet3D
from model.DiNTS import TopologyInstance, DiNTS
from model.Unetpp import BasicUNetPlusPlus
from model.transformer_decoder import ConvBN, kMaXTransformerLayer, kMaXPredictor
from model.TextAttend import TextAttend
from scipy.optimize import linear_sum_assignment

from timm.models.layers import trunc_normal_tf_ as trunc_normal_
from model.sdm import SDM
#from matcher import batch_sigmoid_focal_loss, batch_dice_loss

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
    
class KMamba(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, 
                 cls_organ, 
                 backbone = 'swinunetr', 
                 stage = 'I', 
                 gumbel = False,
                 encoding = 'rand_embedding'):
        # encoding: rand_embedding or word_embedding
        super().__init__()
        self.backbone_name = backbone
        self.stage = stage
        cls_channel = 512
        #out_channels = out_channels + 1
        if backbone == 'swinunetr':
            self.backbone = SwinUNETR(img_size=img_size,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        feature_size=48,
                        drop_rate=0.0,
                        attn_drop_rate=0.0,
                        dropout_path_rate=0.0,
                        use_checkpoint=False,
                        )
            in_dims = [768, 384, 192, 96]
            dims = [384, 192, 96, 48]
            self.kmax_channel = dims
            pixel_out = 48
            projection = pixel_out
            self.post_slayer = SDM(48, 48)
            self.post_dlayer = nn.Sequential(
                ConvBN(96, 48, kernel_size=1, bias=False, norm='3d', act='relu'),
                ConvBN(48, 48, kernel_size=1, bias=False, norm='3d', act='relu'),
                
            )
            if self.stage != 'I':
                self.precls_conv = nn.Sequential(
                    nn.GroupNorm(16, 48),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(48, 8, kernel_size=1)
                )
                self.GAP = nn.Sequential(
                    nn.GroupNorm(16, 768),
                    nn.ReLU(inplace=True),
                    torch.nn.AdaptiveAvgPool3d((1,1,1)),
                    nn.Conv3d(768, 256, kernel_size=1, stride=1, padding=0)
                )
        elif backbone == 'unet':
            self.backbone = UNet3D()
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 64),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 8, kernel_size=1)
            )
            if self.stage != 'I':
                self.precls_conv = nn.Sequential(
                    nn.GroupNorm(16, 64),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(64, 8, kernel_size=1)
                )
                self.GAP = nn.Sequential(
                    nn.GroupNorm(16, 512),
                    nn.ReLU(inplace=True),
                    torch.nn.AdaptiveAvgPool3d((1,1,1)),
                    nn.Conv3d(512, 256, kernel_size=1, stride=1, padding=0)
                )
            in_dims = [512, 256, 128]
            dims = [256, 128, 64]
            pixel_out = 64
            projection = pixel_out
            self.kmax_channel = dims
        elif backbone == 'dints':
            ckpt = torch.load('./model/arch_code_cvpr.pth')
            node_a = ckpt["node_a"]
            arch_code_a = ckpt["arch_code_a"]
            arch_code_c = ckpt["arch_code_c"]

            dints_space = TopologyInstance(
                    channel_mul=1.0,
                    num_blocks=12,
                    num_depths=4,
                    use_downsample=True,
                    arch_code=[arch_code_a, arch_code_c]
                )

            self.backbone = DiNTS(
                    dints_space=dints_space,
                    in_channels=1,
                    num_classes=3,
                    use_downsample=True,
                    node_a=node_a,
                )
            if self.stage != 'I':
                self.precls_conv = nn.Sequential(
                    nn.GroupNorm(16, 32),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(32, 8, kernel_size=1)
                )
                self.GAP = nn.Sequential(
                    nn.GroupNorm(16, 512),
                    nn.ReLU(inplace=True),
                    torch.nn.AdaptiveAvgPool3d((1,1,1)),
                    nn.Conv3d(512, 256, kernel_size=1, stride=1, padding=0)
                )
        elif backbone == 'unetpp':
            self.backbone = BasicUNetPlusPlus(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 32),
                nn.ReLU(inplace=True),
                nn.Conv3d(32, 8, kernel_size=1)
            )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 256),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(256, 256, kernel_size=1, stride=1, padding=0)
            )
        else:
            raise Exception('{} backbone is not implemented in curretn version'.format(backbone))

        self.encoding = encoding
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        if self.stage == 'I':
            self.organ_embedding = nn.Embedding(out_channels, 512)
        elif self.stage == 'II':
            self.register_buffer('organ_embedding', torch.randn(out_channels, 512))
            self.text_to_vision = nn.Linear(512, 256)

            
        self.out_channels = out_channels
        self.cls_organ = cls_organ
        self.cls_tumor = out_channels - cls_organ
        # learnable query features
        self._cluster_centers = nn.Embedding(cls_channel, out_channels)
        
        trunc_normal_(self._cluster_centers.weight, std=1.0)
        self._class_embedding_projection = ConvBN(cls_channel, 256, kernel_size=1, bias=False, norm='1d', act='gelu', conv_type='1d')
        self._mask_embedding_projection = ConvBN(cls_channel, 256, kernel_size=1, bias=False, norm='1d', act='gelu', conv_type='1d')
        self._diff_embedding_projection = ConvBN(cls_channel, projection, kernel_size=1, bias=False, norm='1d', act='gelu', conv_type='1d')
        self._predcitor = kMaXPredictor(in_channel_pixel=pixel_out,in_channel_query=256, num_classes=cls_organ)

        # kmax transformer decoder
        self._kmax_transformer_layers = nn.ModuleList()
        for index, output_stride in enumerate(self.kmax_channel):
            for _ in range(1):
                #print(output_stride)
                self._kmax_transformer_layers.append(kMaXTransformerLayer(
                    stage=stage,
                    num_classes=out_channels,
                    in_channel_pixel=output_stride,
                    in_channel_query=cls_channel,
                    base_filters=128,
                    num_heads=8,
                    bottleneck_expansion=2,
                    key_expansion=1,
                    value_expansion=2,
                    dims=dims[index],
                    in_dims=in_dims[index]
                    #drop_path_prob=drop_path_prob
                    )
                )
        
        # tumor transformer
        #change for text embedding, full text for 768
        self._projection = nn.Sequential(
            ConvBN(projection, projection, kernel_size=5, groups=projection, padding=2, bias=False,
                                                   norm='3d', act='gelu', conv_init='xavier_uniform'),
            ConvBN(projection, projection, kernel_size=1, bias=False, norm='3d', act='gelu'),
            ConvBN(projection, projection, kernel_size=1, bias=True, norm='3d', act=None)
        )
        self.controller = nn.Linear(512, pixel_out)
        self.softmax = nn.Softmax(dim=1)
        
    def get_sim_logits(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        temperature: float = 100,
    ):
        sim_logits = temperature * image_features.matmul(text_features.transpose(-1,-2))
        log_scale = torch.diagonal(sim_logits, dim1=1, dim2=2)/torch.sum(sim_logits, dim=1)
        return log_scale
    
    def normalize_feature(self, feat: torch.Tensor):
        return feat / feat.norm(dim=-1, keepdim=True)
    
    def load_params(self, model_dict):
        if self.backbone_name == 'swinunetr':
            store_dict = self.backbone.state_dict()
            for key in model_dict.keys():
                if 'out' not in key:
                    store_dict[key] = model_dict[key]

            self.backbone.load_state_dict(store_dict)
            print('Use pretrained weights')
        elif self.backbone_name == 'unet':
            store_dict = self.backbone.state_dict()
            for key in model_dict.keys():
                if 'out_tr' not in key:
                    store_dict[key.replace("module.", "")] = model_dict[key]
            self.backbone.load_state_dict(store_dict)
            print('Use pretrained weights')

    def encoding_task(self, task_id):
        N = task_id.shape[0]
        task_encoding = torch.zeros(size=(N, 7))
        for i in range(N):
            task_encoding[i, task_id[i]]=1
        return task_encoding.cuda()
    
    def forward(self, x_in):
        dec4, feats, out = self.backbone(x_in)
        B, C, H, W, D= out.size()
        N = self.out_channels
        # task_encoding torch.Size([31, 256, 1, 1, 1])
        cluster_centers = self._cluster_centers.weight.unsqueeze(0).repeat(B, 1, 1) # B x C x L
        
        if self.stage == 'I': # =====STAGE II: TEXT -> IMAGE BRANCH
            task_encoding = self.organ_embedding.weight
        elif self.stage == 'II':
            task_encoding = self.organ_embedding
            task_encoding = task_encoding.unsqueeze(0).repeat(B, 1, 1)
            
        # =====STAGE I: KMAX TRANSFORMER
        # dual-path transformer
        current_transformer_idx = 0 
        d_feat = dec4
        for i, feat in enumerate(feats): # multi_scale_features = [dec3, dec2, dec1]
            for _ in range(1):
                cluster_centers, d_feat = self._kmax_transformer_layers[current_transformer_idx](
                        d_feat, pixel_feature=feat, query_feature=cluster_centers
                    )
                current_transformer_idx += 1

        class_embeddings = self._class_embedding_projection(cluster_centers)#.transpose(1,2)
        mask_embeddings = self._mask_embedding_projection(cluster_centers)#.transpose(1,2)
        if self.stage == 'I':
            _, logits_out = self._predcitor(
                class_embeddings=class_embeddings,
                mask_embeddings=mask_embeddings,
                pixel_feature=out,
            )
            #logits_out = self.softmax(logits)
        # =====STAGE II: Diffusion Guided            
        elif self.stage == 'II':
            if self.backbone_name == 'swinunetr':
                d_feat = self.up(d_feat)
                out = self.post_slayer(out, d_feat)
                d_feat = self.post_dlayer(torch.cat((d_feat, out), 1))

            d_feat = self._projection(d_feat)
            out = F.normalize(d_feat, p=2, dim=1)
            log_scale = self.get_sim_logits(task_encoding, self.normalize_feature(cluster_centers.transpose(1, 2)))
            
            #cluster_centers[torch.arange(cluster_centers.shape[1]), sim_logits.argmax(dim=-1)]
            
            class_embeddings = self._diff_embedding_projection(cluster_centers)
            #weights = self.controller(class_embeddings).permute(0,2,1)
            logits = out.flatten(start_dim=2, end_dim=4).transpose(1, 2) @ class_embeddings
            logits = torch.einsum('bln,bn->bln', logits, log_scale)
        
            logits_out = logits.transpose(1, 2).reshape(B, N, H, W, D)
            #logits_out = self.softmax(logits_out)
        
        return logits_out