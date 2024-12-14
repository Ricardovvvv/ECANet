# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, ModuleList, Sequential
from torch import Tensor


class DAPPM(BaseModule):
    """DAPPM module in `DDRNet <https://arxiv.org/abs/2101.06085>`_.

    Args:
        in_channels (int): Input channels.
        branch_channels (int): Branch channels.
        out_channels (int): Output channels.
        num_scales (int): Number of scales.
        kernel_sizes (list[int]): Kernel sizes of each scale.
        strides (list[int]): Strides of each scale.
        paddings (list[int]): Paddings of each scale.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU', inplace=True).
        conv_cfg (dict): Config dict for convolution layer in ConvModule.
            Default: dict(order=('norm', 'act', 'conv'), bias=False).
        upsample_mode (str): Upsample mode. Default: 'bilinear'.
    """

    def __init__(self,
                 in_channels: int,
                 branch_channels: int,
                 out_channels: int,
                 num_scales: int,
                 kernel_sizes: List[int] = [5, 9, 17],
                 strides: List[int] = [2, 4, 8],
                 paddings: List[int] = [2, 4, 8],
                 norm_cfg: Dict = dict(type='BN', momentum=0.1),
                 act_cfg: Dict = dict(type='ReLU', inplace=True),
                 conv_cfg: Dict = dict(
                     order=('norm', 'act', 'conv'), bias=False),
                 upsample_mode: str = 'bilinear'):
        super().__init__()

        self.num_scales = num_scales
        self.unsample_mode = upsample_mode
        self.in_channels = in_channels
        self.branch_channels = branch_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.conv_cfg = conv_cfg

        self.scales = ModuleList([
            ConvModule(
                in_channels,
                branch_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                **conv_cfg)
        ])
        for i in range(1, num_scales - 1):
            self.scales.append(
                Sequential(*[
                    nn.AvgPool2d(
                        kernel_size=kernel_sizes[i - 1],
                        stride=strides[i - 1],
                        padding=paddings[i - 1]),
                    ConvModule(
                        in_channels,
                        branch_channels,
                        kernel_size=1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        **conv_cfg)
                ]))
        self.scales.append(
            Sequential(*[
                nn.AdaptiveAvgPool2d((1, 1)),
                ConvModule(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **conv_cfg)
            ]))
        self.processes = ModuleList()
        for i in range(num_scales - 1):
            self.processes.append(
                ConvModule(
                    branch_channels,
                    branch_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    **conv_cfg))

        self.compression = ConvModule(
            branch_channels * num_scales,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **conv_cfg)

        self.shortcut = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **conv_cfg)

    def forward(self, inputs: Tensor):
        feats = []
        feats.append(self.scales[0](inputs))

        for i in range(1, self.num_scales):
            feat_up = F.interpolate(
                self.scales[i](inputs),
                size=inputs.shape[2:],
                mode=self.unsample_mode)
            feats.append(self.processes[i - 1](feat_up + feats[i - 1]))

        return self.compression(torch.cat(feats,
                                          dim=1)) + self.shortcut(inputs)


class PAPPM(DAPPM):
    """PAPPM module in `PIDNet <https://arxiv.org/abs/2206.02066>`_.

    Args:
        in_channels (int): Input channels.
        branch_channels (int): Branch channels.
        out_channels (int): Output channels.
        num_scales (int): Number of scales.
        kernel_sizes (list[int]): Kernel sizes of each scale.
        strides (list[int]): Strides of each scale.
        paddings (list[int]): Paddings of each scale.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN', momentum=0.1).
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU', inplace=True).
        conv_cfg (dict): Config dict for convolution layer in ConvModule.
            Default: dict(order=('norm', 'act', 'conv'), bias=False).
        upsample_mode (str): Upsample mode. Default: 'bilinear'.
    """

    def __init__(self,
                 in_channels: int,
                 branch_channels: int,
                 out_channels: int,
                 num_scales: int,
                 kernel_sizes: List[int] = [5, 9, 17],
                 strides: List[int] = [2, 4, 8],
                 paddings: List[int] = [2, 4, 8],
                 norm_cfg: Dict = dict(type='BN', momentum=0.1),
                 act_cfg: Dict = dict(type='ReLU', inplace=True),
                 conv_cfg: Dict = dict(
                     order=('norm', 'act', 'conv'), bias=False),
                 upsample_mode: str = 'bilinear'):
        super().__init__(in_channels, branch_channels, out_channels,
                         num_scales, kernel_sizes, strides, paddings, norm_cfg,
                         act_cfg, conv_cfg, upsample_mode)

        self.processes = ConvModule(
            self.branch_channels * (self.num_scales - 1),
            self.branch_channels * (self.num_scales - 1),
            kernel_size=3,
            padding=1,
            groups=self.num_scales - 1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            **self.conv_cfg)

    def forward(self, inputs: Tensor):
        x_ = self.scales[0](inputs)
        feats = []
        for i in range(1, self.num_scales):
            feat_up = F.interpolate(
                self.scales[i](inputs),
                size=inputs.shape[2:],
                mode=self.unsample_mode,
                align_corners=False)
            feats.append(feat_up + x_)
        scale_out = self.processes(torch.cat(feats, dim=1))
        return self.compression(torch.cat([x_, scale_out],
                                          dim=1)) + self.shortcut(inputs)


class SCE(nn.Module):
    def __init__(self, in_channels = 512, head_channels=128,out_channels= 128 , grids=(6, 3, 2, 1)):
        super(SCE, self).__init__()

        self.reduce_channel = ConvModule(in_channels, head_channels,1,1,0)
        self.grids = grids
        self.spp = nn.Sequential()
        self.spp.add_module('spp_1', ConvModule(head_channels, head_channels,1,1,0))
        self.spp.add_module('spp_2', ConvModule(head_channels, head_channels,1,1,0))
        self.spp.add_module('spp_3', ConvModule(head_channels, head_channels,1,1,0))
        self.spp.add_module('spp_4', ConvModule(head_channels, head_channels,1,1,0))

        self.upsampling_method = lambda x, size: F.interpolate(x, size, mode='nearest')

        self.spatial_attention = nn.Sequential(
            ConvModule(head_channels * 4, head_channels, 1, 1, 0),
            nn.Conv2d(head_channels, 4, kernel_size=1, bias=False), ##
            nn.Sigmoid()
        )
        self.out_channels = ConvModule(head_channels,out_channels,1,1,0)

        self.keras_init_weight()
        self.spatial_attention[1].weight.data.zero_()


    def forward(self, x):

        size = x.size()[2:]

        ar = size[1] / size[0]
        x = self.reduce_channel(x) # ??

        context = []
        for i in range(len(self.grids)):
            grid_size = (self.grids[i], max(1, round(ar * self.grids[i])))
            # grid_size = (self.grids[i], self.grids[i])
            x_pooled = F.adaptive_avg_pool2d(x, grid_size)
            x_pooled = self.spp[i].forward(x_pooled)
            x_pooled = self.upsampling_method(x_pooled,size)
            context.append(x_pooled)
            # out = out + x_pooled

        spatial_att = self.spatial_attention(torch.cat(context,dim=1))  + 1 ## truple 4

        x = x + context[0] * spatial_att[:, 0:1, :, :] + context[1] * spatial_att[:, 1:2, :, :]  \
            + context[2] * spatial_att[:, 2:3, :, :] + context[3] * spatial_att[:, 3:4, :, :]

        x = self.out_channels(x)
        return x
    def keras_init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.xavier_normal_(ly.weight)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class DPPM(nn.Module):
    def __init__(self,):
        super(DPPM, self).__init__()

    def forward(self, x):
        
        return x










