#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 15:04:06 2022

@author: leeh43
"""

from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from typing import Union
import torch.nn.functional as F
from lib.utils.tools.logger import Logger as Log
from lib.models.tools.module_helper import ModuleHelper
from timm.models.layers import trunc_normal_tf_ as trunc_normal_

from uxnet_encoder import uxnet_conv
from transformer_decoder import ConvBN, kMaXTransformerLayer, kMaXPredictor

from torch.cuda.amp import autocast

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchbn'):
        super(ProjectionHead, self).__init__()

        Log.info('proj_dim: {}'.format(proj_dim))

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv3d(dim_in, dim_in, kernel_size=1),
                ModuleHelper.BNReLU(dim_in, bn_type=bn_type),
                nn.Conv3d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)

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

# class ResBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self,
#                  in_planes: int,
#                  planes: int,
#                  spatial_dims: int = 3,
#                  stride: int = 1,
#                  downsample: Union[nn.Module, partial, None] = None,
#     ) -> None:
#         """
#         Args:
#             in_planes: number of input channels.
#             planes: number of output channels.
#             spatial_dims: number of spatial dimensions of the input image.
#             stride: stride to use for first conv layer.
#             downsample: which downsample layer to use.
#         """
#
#         super().__init__()
#
#         conv_type: Callable = Conv[Conv.CONV, spatial_dims]
#         norm_type: Callable = Norm[Norm.BATCH, spatial_dims]
#
#         self.conv1 = conv_type(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
#         self.bn1 = norm_type(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv_type(planes, planes, kernel_size=3, padding=1, bias=False)
#         self.bn2 = norm_type(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x:torch.Tensor) -> torch.Tensor:
#         residual = x
#
#         out: torch.Tensor = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out


class DKUNET(nn.Module):

    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        num_queries=128,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
        use_checkpoint=False,
        use_qdloss=False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.
            spatial_dims: number of spatial dims.

        """

        super().__init__()

        # in_channels: int,
        # out_channels: int,
        # img_size: Union[Sequence[int], int],
        # feature_size: int = 16,
        # if not (0 <= dropout_rate <= 1):
        #     raise ValueError("dropout_rate should be between 0 and 1.")
        #
        # if hidden_size % num_heads != 0:
        #     raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.use_checkpoint = use_checkpoint
        # self.feature_size = feature_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        self.use_ = use_qdloss

        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.spatial_dims = spatial_dims

        # self.classification = False
        # self.vit = ViT(
        #     in_channels=in_channels,
        #     img_size=img_size,
        #     patch_size=self.patch_size,
        #     hidden_size=hidden_size,
        #     mlp_dim=mlp_dim,
        #     num_layers=self.num_layers,
        #     num_heads=num_heads,
        #     pos_embed=pos_embed,
        #     classification=self.classification,
        #     dropout_rate=dropout_rate,
        #     spatial_dims=spatial_dims,
        # )
        self.uxnet_3d = uxnet_conv(
            in_chans= self.in_chans,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self._kmax_transformer_layers = nn.ModuleList()

        for index, output_stride in enumerate([384, 192, 96]):
            for _ in range(2):
                self._kmax_transformer_layers.append(kMaXTransformerLayer(
                    num_classes=32,
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

        # class embedding
        self._class_embedding_projection = ConvBN(256, 256, kernel_size=1, bias=False, norm='1d', act='gelu',
                                                  conv_type='1d')
        self._mask_embedding_projection = ConvBN(256, 256, kernel_size=1, bias=False, norm='1d', act='gelu',
                                                  conv_type='1d')
        
        self._predcitor = kMaXPredictor(in_channel_pixel=feat_size[0],
            in_channel_query=256, num_classes=132+1)

        # MLP Net
        self.mlp = MLP(num_queries * 256, 64, num_queries * out_chans) # N x C -> N x K

        # softmax
        self.softmax = nn.Softmax(dim=1)

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)
        # self.conv_proj = ProjectionHead(dim_in=hidden_size)

        # init query space
        self._num_queries = num_queries
        # learnable query features
        self._cluster_centers = nn.Embedding(256, num_queries)
        trunc_normal_(self._cluster_centers.weight, std=1.0)

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x
    
    def forward(self, x_in):
        # in features
        outs = self.uxnet_3d(x_in)
        # print(outs[0].size())
        # print(outs[1].size())
        # print(outs[2].size())
        # print(outs[3].size())
        
        enc1 = self.encoder1(x_in)
        # print(enc1.size())
        # query space init
        B = enc1.shape[0]
        cluster_centers = self._cluster_centers.weight.unsqueeze(0).repeat(B, 1, 1) # B x C x L
        
        if self.use_checkpoint == False:
            enc2 = self.encoder2(outs[0])
            enc3 = self.encoder3(outs[1])
            enc4 = self.encoder4(outs[2])
            # dec4 = self.proj_feat(outs[3], self.hidden_size, self.feat_size)
            enc_hidden = self.encoder5(outs[3])
        else:
            enc2 = checkpoint(self.encoder2,outs[0])
            enc3 = checkpoint(self.encoder3,outs[1])
            enc4 = checkpoint(self.encoder4,outs[2])
            # dec4 = self.proj_feat(outs[3], self.hidden_size, self.feat_size)
            enc_hidden = checkpoint(self.encoder5,outs[3])

        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)

        del enc3
        dec1 = self.decoder3(dec2, enc2)
        del enc2
        dec0 = self.decoder2(dec1, enc1)
        del enc1
        # out = self.decoder1(dec0)

        multi_scale_features = [dec3,dec2] # dec1
        # panoptic_features = [dec0]
        # semantic_feature = [dec3,dec1,dec0]
        # feat = self.conv_proj(dec4)

        current_transformer_idx = 0

        #predictions_class = []
        #predictions_mask = []
        #predictions_pixel_feature = []

        # dual-path transformer
        for i, feat in enumerate(multi_scale_features): # multi_scale_features = [dec3,dec2,dec1]
            for _ in range(2):
                cluster_centers, _ = self._kmax_transformer_layers[current_transformer_idx](
                        pixel_feature=feat, query_feature=cluster_centers
                    )
                #predictions_class.append(prediction_result['class_logits'])
                #predictions_mask.append(prediction_result['mask_logits'])
                #predictions_pixel_feature.append(prediction_result['pixel_feature'])
                current_transformer_idx += 1

        class_embeddings = self._class_embedding_projection(cluster_centers)
        mask_embeddings = self._mask_embedding_projection(cluster_centers)

        # Final predictions.
        prediction_result = self._predcitor(
            class_embeddings=class_embeddings,
            mask_embeddings=mask_embeddings,
            pixel_feature=dec0,
        )
        #predictions_class.append(prediction_result['class_logits'])
        #predictions_mask.append(prediction_result['mask_logits'])
        #predictions_pixel_feature.append(prediction_result['pixel_feature'])

        out0 = {
            'pred_logits': prediction_result['class_logits'].float(),
            'pred_masks': prediction_result['mask_logits'].float(),
            #'pixel_feature': prediction_result['pixel_feature'].float(),
            # 'aux_outputs': self._set_aux_loss(predictions_class, predictions_mask, predictions_pixel_feature),      
        }
        
        #for output_key in ["pixel_feature", "pred_masks", "pred_logits", "aux_semantic_pred"]:
        #    if output_key in out0:
        #        out0[output_key] = out0[output_key].float()

        bs, num_queries = out0["pred_logits"].shape[:2]

        # MLP
        cluster_classification = self.mlp(cluster_centers)
        cluster_classification = cluster_classification.view(B, num_queries, -1)

        # Softmax
        query_response = self.softmax(out0['pred_masks'])
        # Merge
        merge1 = torch.sum(query_response[:, :16, :, :, :], dim=1).unsqueeze(1)
        merge2 = torch.sum(query_response[:, 16:20, :, :, :], dim=1).unsqueeze(1)
        merge3 = torch.sum(query_response[:, 20:, :, :, :], dim=1).unsqueeze(1)

        cluster_assignment = torch.cat((merge1,merge2,merge3), dim=1)
        query_logits = torch.einsum('bnk,bnhwd->bkhwd', cluster_classification, query_response)

        # self.out(self.decoder1(dec0))
        
        return (query_logits, cluster_assignment)