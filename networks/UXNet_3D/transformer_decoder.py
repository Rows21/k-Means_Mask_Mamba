from typing import List
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_tf_ as trunc_normal_


import math

class MyBatchNorm_4d(nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MyBatchNorm_4d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
    
    def _check_input_dim(self, input):
        if input.dim() != 6:
            raise ValueError('expected 6D input (got {}D input)'
                             .format(input.dim()))
    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3, 4, 5])
            # use biased var in train
            var = input.var([0, 2, 3, 4, 5], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None, None, None]) / (torch.sqrt(var[None, :, None, None, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None, None, None] + self.bias[None, :, None, None, None, None]

        return input

def get_activation(name):
    if name is None or name.lower() == 'none':
        return nn.Identity()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'gelu':
        return nn.GELU()

def get_norm(name, channels):
    if name is None or name.lower() == 'none':
        return nn.Identity()
    
    if name.lower() == '1d':
        return nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
    
    if name.lower() == '2d':
        return nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)

    if name.lower() == '3d':
        return nn.BatchNorm3d(channels, eps=1e-3, momentum=0.01)
    
    if name.lower() == '4d':
        return MyBatchNorm_4d(channels, eps=1e-3, momentum=0.01)
        
    if name.lower() == 'syncbn':
        return nn.SyncBatchNorm(channels, eps=1e-3, momentum=0.01)
    
class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, norm=None, act=None,
                 conv_type='3d', conv_init='he_normal', norm_init=1.0):
        super().__init__()
        
        if conv_type == '3d':
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        elif conv_type == '2d':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        elif conv_type == '1d':
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.norm = get_norm(norm, out_channels)
        self.act = get_activation(act)

        if conv_init == 'normal':
            nn.init.normal_(self.conv.weight, std=.02)
        elif conv_init == 'trunc_normal':
            trunc_normal_(self.conv.weight, std=.02)
        elif conv_init == 'he_normal':
            # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal
            trunc_normal_(self.conv.weight, std=math.sqrt(2.0 / in_channels))
        elif conv_init == 'xavier_uniform':
            nn.init.xavier_uniform_(self.conv.weight)
        if bias:
            nn.init.zeros_(self.conv.bias)

        if norm is not None:
            nn.init.constant_(self.norm.weight, norm_init)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
# https://github.com/google-research/deeplab2/blob/7a01a7165e97b3325ad7ea9b6bcc02d67fecd07a/model/decoder/max_deeplab.py#L60
def add_bias_towards_void(query_class_logits, void_prior_prob=0.9):
    class_logits_shape = query_class_logits.shape
    init_bias = [0.0] * class_logits_shape[-1]
    init_bias[-1] = math.log(
      (class_logits_shape[-1] - 1) * void_prior_prob / (1 - void_prior_prob))
    return query_class_logits + torch.tensor(init_bias, dtype=query_class_logits.dtype).to(query_class_logits)

# https://github.com/google-research/deeplab2/blob/main/model/kmax_deeplab.py#L32
class kMaXPredictor(nn.Module):
    def __init__(self, in_channel_pixel, in_channel_query, num_classes=32+1):
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
        pixel_space_feature = self._pixel_space_head_conv1bnact(pixel_space_feature)
        pixel_space_feature = self._pixel_space_head_last_convbn(pixel_space_feature)
        pixel_space_normalized_feature = F.normalize(pixel_space_feature, p=2, dim=1)

        cluster_class_logits = self._transformer_class_head(class_embeddings).permute(0, 2, 1).contiguous()
        cluster_class_logits = add_bias_towards_void(cluster_class_logits)
        cluster_mask_kernel = self._transformer_mask_head(mask_embeddings)
        mask_logits = torch.einsum('bchwd,bcn->bnhwd',
          pixel_space_normalized_feature, cluster_mask_kernel)
        
        mask_logits = self._pixel_space_mask_batch_norm(mask_logits.unsqueeze(dim=1)).squeeze(dim=1) # BN 6D Norm

        return {
            'class_logits': cluster_class_logits, # Cluster Centers BCN
            'mask_logits': mask_logits, # Query Response BNHWD
            'pixel_feature': pixel_space_normalized_feature} # Pixel feature BCHWD
    
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
        num_classes=32,
        in_channel_pixel=2048,
        in_channel_query=256,
        base_filters=128,
        num_heads=8,
        bottleneck_expansion=2,
        key_expansion=1,
        value_expansion=2,
        drop_path_prob=0.0,
    ):
        super().__init__()

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
                                          norm='1d', act='gelu', conv_type='1d')

        self._pixel_conv1_bn_act = ConvBN(in_channel_pixel, self._bottleneck_channels, kernel_size=1, bias=False,
                                          norm='3d', act='gelu')

        self._query_qkv_conv_bn = ConvBN(self._bottleneck_channels, self._total_key_depth * 2 + self._total_value_depth, kernel_size=1, bias=False,
                                          norm='1d', act=None, conv_type='1d')
        trunc_normal_(self._query_qkv_conv_bn.conv.weight, std=initialization_std)

        self._pixel_v_conv_bn = ConvBN(self._bottleneck_channels, self._total_value_depth, kernel_size=1, bias=False,
                                          norm='3d', act=None)
        trunc_normal_(self._pixel_v_conv_bn.conv.weight, std=initialization_std)

        self._query_self_attention = AttentionOperation(channels_v=self._total_value_depth, num_heads=num_heads)

        self._query_conv3_bn = ConvBN(self._total_value_depth, in_channel_query, kernel_size=1, bias=False,
                                          norm='syncbn', act=None, conv_type='1d', norm_init=0.0)

        self._query_ffn_conv1_bn_act = ConvBN(in_channel_query, 2048, kernel_size=1, bias=False,
                                          norm='1d', act='gelu', conv_type='1d')
        self._query_ffn_conv2_bn = ConvBN(2048, in_channel_query, kernel_size=1, bias=False,
                                          norm='1d', act=None, conv_type='1d', norm_init=0.0)

        self._predcitor = kMaXPredictor(in_channel_pixel=self._bottleneck_channels,
            in_channel_query=self._bottleneck_channels, num_classes=num_classes)
        self._kmeans_query_batch_norm_retrieved_value = get_norm('1d', self._total_value_depth)
        self._kmeans_query_conv3_bn = ConvBN(self._total_value_depth, in_channel_query, kernel_size=1, bias=False,
                                          norm='1d', act=None, conv_type='1d', norm_init=0.0)


    def forward(self, pixel_feature, query_feature):
        N, C, H, W, D = pixel_feature.shape
        _, Z, L = query_feature.shape
        pixel_space = self._pixel_conv1_bn_act(F.gelu(pixel_feature)) # N C H W D
        query_space = self._query_conv1_bn_act(query_feature) # N x C x L

        # k-means cross-attention.
        pixel_value = self._pixel_v_conv_bn(pixel_space) # N C H W D
        pixel_value = pixel_value.reshape(N, self._total_value_depth, H*W*D)
        # k-means assignment.
        prediction_result = self._predcitor(
            mask_embeddings=query_space, class_embeddings=query_space, pixel_feature=pixel_space)
        
        with torch.no_grad():
            clustering_result = prediction_result['mask_logits'].flatten(2).detach() # N L HW
            index = clustering_result.max(1, keepdim=True)[1]
            clustering_result = torch.zeros_like(clustering_result, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)

        with autocast(enabled=False):
        # k-means update.
            kmeans_update = torch.einsum('blm,bdm->bdl', clustering_result.float(), pixel_value.float()) # N x C x L

        kmeans_update = self._kmeans_query_batch_norm_retrieved_value(kmeans_update)
        kmeans_update = self._kmeans_query_conv3_bn(kmeans_update)
        query_feature = query_feature + self.drop_path_kmeans(kmeans_update)

        # query self-attention.
        query_qkv = self._query_qkv_conv_bn(query_space)
        query_q, query_k, query_v = torch.split(query_qkv,
         [self._total_key_depth, self._total_key_depth, self._total_value_depth], dim=1)
        query_q = query_q.reshape(N, self._num_heads, self._total_key_depth//self._num_heads, L)
        query_k = query_k.reshape(N, self._num_heads, self._total_key_depth//self._num_heads, L)
        query_v = query_v.reshape(N, self._num_heads, self._total_value_depth//self._num_heads, L)
        self_attn_update = self._query_self_attention(query_q, query_k, query_v)
        self_attn_update = self._query_conv3_bn(self_attn_update)
        query_feature = query_feature + self.drop_path_attn(self_attn_update)
        query_feature = F.gelu(query_feature)

        # FFN.
        ffn_update = self._query_ffn_conv1_bn_act(query_feature)
        ffn_update = self._query_ffn_conv2_bn(ffn_update)
        query_feature = query_feature + self.drop_path_ffn(ffn_update)
        query_feature = F.gelu(query_feature)

        return query_feature, prediction_result