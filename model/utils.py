import math
import torch
from torch import nn
from timm.models.layers import trunc_normal_tf_ as trunc_normal_

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