from typing import Sequence, Tuple, Type, Union

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
from timm.models.layers import trunc_normal_tf_ as trunc_normal_



class KMamba(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, backbone = 'swinunetr', encoding = 'rand_embedding'):
        # encoding: rand_embedding or word_embedding
        super().__init__()
        self.backbone_name = backbone
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
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 512),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool3d((1,1,1)),
                nn.Conv3d(512, 256, kernel_size=1, stride=1, padding=0)
            )
            self.kmax_channel = [256,128]
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

        weight_nums, bias_nums = [], []
        weight_nums.append(8*8)
        weight_nums.append(8*8)
        weight_nums.append(8*1)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(1)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.controller = nn.Conv3d(256+256, sum(weight_nums+bias_nums), kernel_size=1, stride=1, padding=0)
        if self.encoding == 'rand_embedding':
            self.organ_embedding = nn.Embedding(out_channels, 256)
        elif self.encoding == 'word_embedding':
            self.register_buffer('organ_embedding', torch.randn(out_channels, 512))
            self.text_to_vision = nn.Linear(512, 256)
        self.class_num = out_channels

        # learnable query features
        self._cluster_centers = nn.Embedding(256, out_channels)
        trunc_normal_(self._cluster_centers.weight, std=1.0)

        # kmax transformer decoder
        self._kmax_transformer_layers = nn.ModuleList()
        for index, output_stride in enumerate(self.kmax_channel):
            for _ in range(2):
                print(output_stride)
                self._kmax_transformer_layers.append(kMaXTransformerLayer(
                    num_classes=out_channels,
                    in_channel_pixel=output_stride,
                    in_channel_query=256,
                    base_filters=128,
                    num_heads=8,
                    bottleneck_expansion=2,
                    key_expansion=1,
                    value_expansion=2
                    #drop_path_prob=drop_path_prob
                    )
                )

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

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 1)
            # print(weight_splits[l].shape, bias_splits[l].shape)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 5
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            # print(i, x.shape, w.shape)
            x = F.conv3d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def forward(self, x_in):
        dec4, outs = self.backbone(x_in)
        # =================================================================================
        if self.encoding == 'rand_embedding':
            task_encoding = self.organ_embedding.weight.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        elif self.encoding == 'word_embedding':
            task_encoding = F.relu(self.text_to_vision(self.organ_embedding))
            task_encoding = task_encoding.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        # task_encoding torch.Size([31, 256, 1, 1, 1])
        B=2
        cluster_centers = self._cluster_centers.weight.unsqueeze(0).repeat(B, 1, 1) # B x C x L
        
        # dual-path transformer
        current_transformer_idx = 0 
        for i, feat in enumerate(outs[:-1]): # multi_scale_features = [dec3,dec2,dec1]
            
            for _ in range(2):
                cluster_centers, _ = self._kmax_transformer_layers[current_transformer_idx](
                        pixel_feature=feat, query_feature=cluster_centers
                    )
                #predictions_class.append(prediction_result['class_logits'])
                #predictions_mask.append(prediction_result['mask_logits'])
                #predictions_pixel_feature.append(prediction_result['pixel_feature'])
                current_transformer_idx += 1

        x_feat = self.GAP(dec4)
        B = x_feat.shape[0]
        logits_array = []
        for i in range(B):
            x_cond = torch.cat([x_feat[i].unsqueeze(0).repeat(self.class_num,1,1,1,1), task_encoding], 1)
            params = self.controller(x_cond)
            params.squeeze_(-1).squeeze_(-1).squeeze_(-1) # 32 * 153
            
            head_inputs = self.precls_conv(outs[-1][i].unsqueeze(0)) # 1 8 96 96 96
            head_inputs = head_inputs.repeat(self.class_num,1,1,1,1) # 32 8 96 96 96 
            N, _, D, H, W = head_inputs.size()
            head_inputs = head_inputs.reshape(1, -1, D, H, W) # 1 256 96 96 96  
            # print(head_inputs.shape, params.shape)
            weights, biases = self.parse_dynamic_params(params, 8, self.weight_nums, self.bias_nums)

            logits = self.heads_forward(head_inputs, weights, biases, N)
            logits_array.append(logits.reshape(1, -1, D, H, W))
        
        out = torch.cat(logits_array,dim=0) # 2 32 96 96 96 
        # print(out.shape)
        return out