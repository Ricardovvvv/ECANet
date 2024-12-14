# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple, Union

import torch.nn as nn
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from torch import Tensor

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType, SampleList


@MODELS.register_module()
class ECANetHead(BaseDecodeHead):
    """Decode head for ECANet."""
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 **kwargs):
        super().__init__(
            in_channels,
            channels,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

        self.head = self._make_base_head(self.in_channels, self.channels)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
            self,
            inputs: Union[Tensor,
                          Tuple[Tensor]]) -> Union[Tensor, Tuple[Tensor]]:
        if self.training:
            temp0,temp1,temp2,x_c = inputs
            x_c = self.head(x_c)
            x_c = self.cls_seg(x_c)
            return x_c, temp0, temp1, temp2
        else:
            x_c = self.head(inputs)
            x_c = self.cls_seg(x_c)
            return x_c

    def _make_base_head(self, in_channels: int,
                        channels: int) -> nn.Sequential:
        layers = [
            ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                order=('norm', 'act', 'conv')),
            build_norm_layer(self.norm_cfg, channels)[1],
            build_activation_layer(self.act_cfg),
        ]

        return nn.Sequential(*layers)



    def loss_by_feat(self, seg_logits: Tuple[Tensor],
                     batch_data_samples: SampleList) -> dict:
        loss = dict()
        context_logit, spatial_logit_16, spatial_logit_32, spatial_logit_64 = seg_logits
        seg_label = self._stack_batch_gt(batch_data_samples)

        context_logit = resize(
            context_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        spatial_logit_16 = resize(
            spatial_logit_16,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        spatial_logit_32 = resize(
            spatial_logit_32,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        spatial_logit_64 = resize(
            spatial_logit_64,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        seg_label = seg_label.squeeze(1)

        loss['loss_context'] = self.loss_decode[0](context_logit, seg_label)
        loss['loss_spatial_16'] = self.loss_decode[1](spatial_logit_16, seg_label)
        loss['loss_spatial_32'] = self.loss_decode[2](spatial_logit_32, seg_label)
        loss['loss_spatial_64'] = self.loss_decode[3](spatial_logit_64, seg_label)

        return loss
