from typing import List
import torch

from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_tf_ as trunc_normal_
from model.mambablock import MambaLayer
import math
from model.TextAttend import TextAttend
from model.sdm import SDM
from model.utils import ConvBN, get_norm

# https://github.com/google-research/deeplab2/blob/7a01a7165e97b3325ad7ea9b6bcc02d67fecd07a/model/decoder/max_deeplab.py#L60
def add_bias_towards_void(query_class_logits, void_prior_prob=0.9):
    class_logits_shape = query_class_logits.shape
    init_bias = [0.0] * class_logits_shape[-1]
    init_bias[-1] = math.log(
      (class_logits_shape[-1] - 1) * void_prior_prob / (1 - void_prior_prob))
    return query_class_logits + torch.tensor(init_bias, dtype=query_class_logits.dtype).to(query_class_logits)

# https://github.com/google-research/deeplab2/blob/main/model/kmax_deeplab.py#L32
class kMaXPredictor(nn.Module):
    def __init__(self, in_channel_pixel, 
                 in_channel_query, 
                 num_classes=32+1, 
                 ):
        super().__init__()
        
        self._pixel_space_head_conv0bnact = ConvBN(in_channel_pixel, in_channel_pixel, kernel_size=5, groups=in_channel_pixel, padding=2, bias=False,
                                                   norm='3d', act='gelu', conv_init='xavier_uniform')
        self._pixel_space_head_conv1bnact = ConvBN(in_channel_pixel, 256, kernel_size=1, bias=False, norm='3d', act='gelu')
        self._pixel_space_head_last_convbn = ConvBN(256, 128, kernel_size=1, bias=True, norm='3d', act=None)
        trunc_normal_(self._pixel_space_head_last_convbn.conv.weight, std=0.01)

        self._transformer_mask_head = ConvBN(256, 128, kernel_size=1, bias=False, norm='1d', act=None, conv_type='1d')
        self._transformer_class_head = ConvBN(256, num_classes, kernel_size=1, norm=None, act=None, conv_type='1d')
        trunc_normal_(self._transformer_class_head.conv.weight, std=0.01)

        self._pixel_space_mask_batch_norm = get_norm('4d', channels=1)
        nn.init.constant_(self._pixel_space_mask_batch_norm.weight, 0.1)

    def forward(self, mask_embeddings, class_embeddings, pixel_feature):
        # mask_embeddings/class_embeddings: B x C x N
        # pixel feature: B x C x H x W x D
        pixel_space_feature = self._pixel_space_head_conv0bnact(pixel_feature)
        pixel_out_feature = self._pixel_space_head_conv1bnact(pixel_space_feature)
        pixel_space_feature = self._pixel_space_head_last_convbn(pixel_out_feature) # 256 -> 128
        pixel_space_normalized_feature = F.normalize(pixel_space_feature, p=2, dim=1)

        cluster_class_logits = self._transformer_class_head(class_embeddings).permute(0, 2, 1).contiguous()
        cluster_class_logits = add_bias_towards_void(cluster_class_logits)
        cluster_mask_kernel = self._transformer_mask_head(mask_embeddings)
        #print(cluster_class_logits.shape)
        mask_logits = torch.einsum('bchwd,bcn->bnhwd',
          pixel_space_normalized_feature, cluster_mask_kernel)
        
        mask_logits = self._pixel_space_mask_batch_norm(mask_logits.unsqueeze(dim=1)).squeeze(dim=1) # BN 6D Norm

        return cluster_class_logits, mask_logits# Query Response BNHWD
    
# https://github.com/google-research/deeplab2/blob/7a01a7165e97b3325ad7ea9b6bcc02d67fecd07a/model/layers/dual_path_transformer.py#L41
class AttentionOperation(nn.Module):
    def __init__(self, channels_v, num_heads):
        super().__init__()
        self._batch_norm_similarity = get_norm('2d', num_heads)
        self._batch_norm_retrieved_value = get_norm('1d', channels_v)

    def forward(self, query, key, value):
        N, _, _, L = query.shape
        _, num_heads, C, _ = value.shape
        similarity_logits = torch.einsum('bhdl,bhdm->bhlm', query, key)
        similarity_logits = self._batch_norm_similarity(similarity_logits)

        with autocast(enabled=False):
            attention_weights = F.softmax(similarity_logits.float(), dim=-1)
        retrieved_value = torch.einsum(
            'bhlm,bhdm->bhdl', attention_weights, value)
        retrieved_value = retrieved_value.reshape(N, num_heads * C, L)
        retrieved_value = self._batch_norm_retrieved_value(
            retrieved_value)
        retrieved_value = F.gelu(retrieved_value)
        return retrieved_value
    
# https://github.com/google-research/deeplab2/blob/7a01a7165e97b3325ad7ea9b6bcc02d67fecd07a/model/layers/dual_path_transformer.py#L107
class kMaXTransformerLayer(nn.Module):
    def __init__(
        self,
        stage,
        num_classes=32,
        in_channel_pixel=2048,
        in_channel_query=256,
        base_filters=128,
        num_heads=8,
        bottleneck_expansion=2,
        key_expansion=1,
        value_expansion=2,
        drop_path_prob=0.0,
        dims=96,
        in_dims=48,
    ):
        super().__init__()
        self.stage = stage
        self._num_classes = num_classes
        self._num_heads = num_heads
        self._bottleneck_channels = int(round(base_filters * bottleneck_expansion))
        self._total_key_depth = int(round(base_filters * key_expansion))
        self._total_value_depth = int(round(base_filters * value_expansion))

        # Per tf2 implementation, the same drop path prob are applied to:
        # 1. k-means update for object query
        # 2. self/cross-attetion for object query
        # 3. ffn for object query

        self.drop_path_kmeans = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()
        self.drop_path_attn = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()
        self.drop_path_ffn = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()

        initialization_std = self._bottleneck_channels ** -0.5
        self._query_conv1_bn_act = ConvBN(in_channel_query, self._bottleneck_channels, kernel_size=1, bias=False,
                                          norm='1d', act='relu', conv_type='1d')
        #nn.Linear(in_channel_query, self._bottleneck_channels)
        #nn.Conv1d(in_channels=in_channel_query, out_channels=self._bottleneck_channels, kernel_size=1, stride=1, padding=0, bias=False)
        #ConvBN(in_channel_query, self._bottleneck_channels, kernel_size=1, bias=False,
                                    #      norm='1d', act='relu', conv_type='1d')
                                
        self._pixel_conv1_bn_act = ConvBN(in_channel_pixel, self._bottleneck_channels, kernel_size=1, bias=False,
                                          norm='3d', act='gelu')

        self._query_qkv_conv_bn = ConvBN(self._bottleneck_channels, self._total_key_depth * 2 + self._total_value_depth, kernel_size=1, bias=False,
                                          norm='1d', act=None, conv_type='1d')
        trunc_normal_(self._query_qkv_conv_bn.conv.weight, std=initialization_std)
        #nn.Linear(self._bottleneck_channels, self._total_key_depth * 2 + self._total_value_depth)
        #ConvBN(self._bottleneck_channels, self._total_key_depth * 2 + self._total_value_depth, kernel_size=1, bias=False,
                                          #norm='1d', act=None, conv_type='1d')
        #trunc_normal_(self._query_qkv_conv_bn.conv.weight, std=initialization_std)

        self._pixel_v_conv_bn = ConvBN(self._bottleneck_channels, self._total_value_depth, kernel_size=1, bias=False,
                                          norm='3d', act=None)
        trunc_normal_(self._pixel_v_conv_bn.conv.weight, std=initialization_std)

        self._query_self_attention = AttentionOperation(channels_v=self._total_value_depth, num_heads=num_heads)

        self._query_conv3_bn = ConvBN(self._total_value_depth, in_channel_query, kernel_size=1, bias=False,
                                          norm='1d', act=None, conv_type='1d', norm_init=0.0)
        #nn.Linear(self._total_value_depth, in_channel_query)
                               #ConvBN(self._total_value_depth, in_channel_query, kernel_size=1, bias=False,
                               #           norm='1d', act=None, conv_type='1d', norm_init=0.0)

        self._query_ffn_conv1_bn_act = ConvBN(in_channel_query, 2048, kernel_size=1, bias=False,
                                          norm='1d', act='gelu', conv_type='1d')
        self._query_ffn_conv2_bn = ConvBN(2048, in_channel_query, kernel_size=1, bias=False,
                                          norm='1d', act=None, conv_type='1d', norm_init=0.0)

        self._predcitor = kMaXPredictor(in_channel_pixel=self._bottleneck_channels,
            in_channel_query=self._bottleneck_channels, num_classes=num_classes)
        self._kmeans_query_batch_norm_retrieved_value = get_norm('1d', self._total_value_depth)
        self._kmeans_query_conv3_bn = ConvBN(self._total_value_depth, in_channel_query, kernel_size=1, bias=False,
                                          norm='1d', act=None, conv_type='1d', norm_init=0.0)

        # ssm block init
        self.mamba_layer = MambaLayer(input_dim=num_classes, output_dim=num_classes)
        
        # anomaly layer
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.sdm = SDM(dims, in_dims)
        self.diff = nn.Sequential(
            ConvBN(in_dims + dims, dims, kernel_size=1, bias=False, norm='3d', act='relu'),
            ConvBN(dims, dims, kernel_size=1, bias=False, norm='3d', act='relu'),  
            )
        
        self.anomal_attn = TextAttend(dim=dims, 
                    out_dim=in_channel_query, 
                    num_heads=8, 
                    norm_layer=nn.LayerNorm, 
                    in_features=in_dims, 
                    mlp_ratio=4, 
                    hard=True, 
                    gumbel=True, 
                    sum_assign=False, 
                    assign_eps=1., 
                    gumbel_tau=1.)
        
    def anomaly_score(self, anomaly):
        anomaly_mask = anomaly > 0.5
        anomaly[anomaly_mask] = anomaly[anomaly_mask] * 0.2
        anomaly[~anomaly_mask] = -1.0 # anomaly[~anomaly_mask] * 0.0
        anomaly = anomaly.unsqueeze(1).repeat(1, self._num_classes, 1, 1, 1)
        return anomaly

    def forward(self, d_feat, pixel_feature, query_feature):
        B, C, H, W, D = pixel_feature.shape
        B, Z, L = query_feature.shape
        
        pixel_space = self._pixel_conv1_bn_act(F.gelu(pixel_feature)) # B C HWD
        query_space = self._query_conv1_bn_act(query_feature) # B x C x L #.transpose(1,2)

        # k-means cross-attention.
        pixel_value = self._pixel_v_conv_bn(pixel_space) # B C HWD
        pixel_value = pixel_value.reshape(B, self._total_value_depth, H*W*D)
        # k-means assignment.
        _, mask_logits = self._predcitor(
            mask_embeddings=query_space, class_embeddings=query_space, pixel_feature=pixel_space)
        
        with torch.no_grad():
            clustering_result = mask_logits.flatten(2).detach() # B N HWD
            anomaly_score = -torch.max(mask_logits, dim=1)[0]
            score_min, score_max = anomaly_score.min(), anomaly_score.max()
            anomaly_score_normalized = (anomaly_score - score_min)/(score_max - score_min)
            #mask_logits = mask_logits.permute(0,2,1)
            clustering_result = self.mamba_layer(clustering_result)# + mask_logits # norm_f
            index = clustering_result.max(1, keepdim=True)[1]
            clustering_result = torch.zeros_like(clustering_result, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
            anomaly = self.anomaly_score(anomaly_score_normalized)
            
        with autocast(enabled=False):
        # k-means update.
            kmeans_update = torch.einsum('blm,bdm->bdl', clustering_result.float(), pixel_value.float()) # N x C x L

        kmeans_update = self._kmeans_query_batch_norm_retrieved_value(kmeans_update)
        kmeans_update = self._kmeans_query_conv3_bn(kmeans_update) # .transpose(1,2)
        query_feature = query_feature + self.drop_path_kmeans(kmeans_update)

        # anomaly 
        if self.stage == 'II':
            d_feat = self.up(d_feat)
            feat = self.sdm(pixel_feature, d_feat)
            d_feat = self.diff(torch.cat((d_feat, feat), 1))
            query_feature, _ = self.anomal_attn(feat, query_feature.transpose(1,2), anomaly)
        
        # query self-attention.
        query_qkv = self._query_qkv_conv_bn(query_space) #.transpose(1,2)
        query_q, query_k, query_v = torch.split(query_qkv,
         [self._total_key_depth, self._total_key_depth, self._total_value_depth], dim=1)
        query_q = query_q.reshape(B, self._num_heads, self._total_key_depth//self._num_heads, L)
        query_k = query_k.reshape(B, self._num_heads, self._total_key_depth//self._num_heads, L)
        query_v = query_v.reshape(B, self._num_heads, self._total_value_depth//self._num_heads, L)
        self_attn_update = self._query_self_attention(query_q, query_k, query_v)
        self_attn_update = self._query_conv3_bn(self_attn_update)#.transpose(1,2)

        query_feature = query_feature + self.drop_path_attn(self_attn_update)
        query_feature = F.gelu(query_feature)

        # FFN.
        ffn_update = self._query_ffn_conv1_bn_act(query_feature)
        ffn_update = self._query_ffn_conv2_bn(ffn_update)
        query_feature = query_feature + self.drop_path_ffn(ffn_update)
        query_feature = F.gelu(query_feature)

        return query_feature, d_feat