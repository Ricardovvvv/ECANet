import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.utils import SPPFFM,SPCFFM,DPPM  # the code of three modules will coming soon.
from mmcv.cnn import ConvModule

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1


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
        out = self.pp(x_high, x_low)
        out = self.pc(out)
        return out



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        #self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        #self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        #self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        #self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=True)
        #self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=8):
        super(segmenthead, self).__init__()
        #self.bn1 = BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=True)
        # self.bn2 = BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(x))
        out = self.conv2(self.relu(x))


        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                                size=[height, width],
                                mode='bilinear')

        return out


class DualResNet(nn.Module):

    def __init__(self, block, layers, num_classes=19, planes=64, spp_planes=128, head_planes=128):
        super(DualResNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            # BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            # BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, planes, planes, layers[0])
        self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes * 2, planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes * 4, planes * 8, layers[3], stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)

        self.spp = DPPM(planes*16,spp_planes,planes*16,num_scales=5)

        self.class_feature_k = nn.Conv1d(planes * 2, 19, 1, bias=False)
        self.class_feature_v = nn.Conv1d(19, planes * 2, 1, bias=False)
        self.class_feature_v.weight.data = self.class_feature_k.weight.data.permute(1, 0, 2)

        self.sffm_16 = SFFM(planes*2,planes*4,self.class_feature_k,self.class_feature_v,False)
        self.sffm_32 = SFFM(planes*2,planes*8,self.class_feature_k,self.class_feature_v,False)
        self.sffm_64 = SFFM(planes*2,planes*16,self.class_feature_k,self.class_feature_v,False)

        self.final_layer = segmenthead(planes * 2, head_planes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                # nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        b,_,_,_ = x.shape
        layers = []

        x = self.conv1(x)

        x = self.layer1(x)

        x = self.layer2(self.relu(x))  # 1/8
        layers.append(x)

        x_c = self.layer3(self.relu(x))  # 1/16
        x_s = self.sffm_16(x,x_c)

        x_c = self.layer4(self.relu(x_c))  # 1/32
        x_s = self.sffm_32(x_s,x_c)

        x_c = self.layer5(self.relu(x_c))  # 1/64
        x_c = self.spp(x_c)
        x_s = self.sffm_64(x_s,x_c)

        x_s = self.final_layer(x_s)
        return x_s


if __name__ == '__main__':

    import time

    device = torch.device('cuda:2')
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True

    model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=19, planes=32, spp_planes=128, head_planes=64)
    model.eval()
    model.to(device)
    iterations = None

    input = torch.randn(1, 3, 1024, 2048).cuda('cuda:2')
    with torch.no_grad():
        for _ in range(10):
            model(input)

        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)

        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)

    from thop import profile

    flops, params = profile(model, (input,))

    print('FLOPs: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))






