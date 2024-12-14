import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, build_norm_layer,build_activation_layer
from mmengine.model import BaseModule

from mmseg.models.utils import BasicBlock, Bottleneck, resize,SPCFFM,SPPFFM
from mmseg.models.utils import DAPPM,SCE,PAPPM,PPM,DPPM
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType

class SFFM(nn.Module):
    '''seletive feature fusion module'''
    def __init__(self,high_channels,low_channels,class_feature_k,class_feature_v,
                 align_corners,norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.class_feature_k = class_feature_k
        self.conv_low = nn.Sequential(
            ConvModule(
                low_channels,
                max(low_channels//2,high_channels),
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,),
            ConvModule(
                max(low_channels // 2, high_channels),
                high_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg, ),
        )

        self.conv_high = ConvModule(
                high_channels,
                high_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,)

        self.align_corners = align_corners
        self.pp = SPPFFM(align_corners=align_corners)
        self.pc = SPCFFM(high_channels,class_feature_k,
                                          class_feature_v, norm_cfg,act_cfg)

    def forward(self,x_high,x_low):
        x_high = self.conv_high(x_high)
        x_low = self.conv_low(x_low)

        b,c,h,w = x_low.shape
        segment = self.class_feature_k(x_low.flatten(2))
        segment = segment.reshape(b,-1,h,w)

        out = self.pp(x_high, x_low)
        out = self.pc(out)
        return out, segment


@MODELS.register_module()
class ECANet(BaseModule):

    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 32,
                 ppm_channels: int = 128,
                 num_class: int = 19,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.ppm_channels = ppm_channels
        self.num_class = num_class

        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners

        # stage 0-2
        self.stem = self._make_stem_layer(in_channels, channels, num_blocks=2)
        self.relu = nn.ReLU()

        # low resolution(context) branch
        self.context_branch_layers = nn.ModuleList()
        for i in range(3):
            self.context_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    inplanes=channels * 2**(i + 1),
                    planes=channels * 8 if i > 0 else channels * 4,
                    num_blocks=2 if i < 2 else 1,
                    stride=2))

        self.spp = DPPM(channels*16,ppm_channels,channels*16,num_scales=5)

        self.class_feature_k = nn.Conv1d(channels*2, num_class, 1, bias=False)
        self.class_feature_v = nn.Conv1d(num_class, channels*2, 1, bias=False)
        self.class_feature_v.weight.data = self.class_feature_k.weight.data.permute(1, 0, 2)

        self.sffm_16 = SFFM(channels*2,channels*4,self.class_feature_k,self.class_feature_v,
                            align_corners=align_corners, norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
        self.sffm_32 = SFFM(channels*2,channels*8,self.class_feature_k,self.class_feature_v,
                            align_corners=align_corners, norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
        self.sffm_64 = SFFM(channels*2,channels*16,self.class_feature_k,self.class_feature_v,
                            align_corners=align_corners,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)

    def _make_stem_layer(self, in_channels, channels, num_blocks):
        layers = [
            ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            ConvModule(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        ]

        layers.extend([
            self._make_layer(BasicBlock, channels, channels, num_blocks),
            nn.ReLU(),
            self._make_layer(
                BasicBlock, channels, channels * 2, num_blocks, stride=2),
        ])

        return nn.Sequential(*layers)

    def _make_layer(self, block, inplanes, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, planes * block.expansion)[1])

        layers = [
            block(
                in_channels=inplanes,
                channels=planes,
                stride=stride,
                downsample=downsample)
        ]
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=inplanes,
                    channels=planes,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg_out=None if i == num_blocks - 1 else self.act_cfg))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward function."""

        # stage 0-2
        x = self.stem(x)  # 1/8

        # stage3
        x_c = self.context_branch_layers[0](self.relu(x))  # 1/16
        x_,segment = self.sffm_16(self.relu(x),self.relu(x_c))
        if self.training:
            temp0 = segment

        # stage4
        x_c = self.context_branch_layers[1](self.relu(x_c))  # 1/32
        x_,segment = self.sffm_32(self.relu(x_),self.relu(x_c))
        if self.training:
            temp2 = segment

        # stage5
        x_c = self.context_branch_layers[2](self.relu(x_c))  # 1/64
        x_c = self.spp(x_c)
        x_,segment = self.sffm_64(self.relu(x_),self.relu(x_c))
        if self.training:
            temp3 = segment

        return (temp0,temp2,temp3,x_) if self.training else x_










