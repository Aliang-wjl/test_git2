import torch
from spikingjelly.activation_based import surrogate
from spikingjelly.activation_based import encoding
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from spikingjelly.activation_based import monitor, neuron, functional, layer
import sys
import numpy as np
from torch.cuda import amp
import time
import matplotlib.pyplot as plt
import torch.nn as nn
from torchsummary import summary
import datetime
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
import os



# SEW原始网络
class BasicBlock(nn.Module):
    """
    实现ResNet网络的第一个基础架构
    """
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f=None):
        super(BasicBlock, self).__init__()
        
        self.connect_f = connect_f
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('SpikingBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SpikingBasicBlock")

        self.layer1 = nn.Sequential(
            layer.Conv2d(inplanes, planes, kernel_size=3, stride = stride, padding=1, bias=False),
            norm_layer(planes),
            neuron.IFNode(surrogate_function=surrogate.ATan()))
        
        self.layer2 = nn.Sequential(
            layer.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
            norm_layer(planes),
            neuron.IFNode(surrogate_function=surrogate.ATan()))

        self.downsample = downsample
        
        # 设置所有的层为多步模式
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        identity = x
        out = self.layer1(x)
        out = self.layer2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
#         print(out.shape)
#         print(identity.shape)
        if self.connect_f == 'ADD':
            out += identity
        elif self.connect_f == 'AND':
            out *= identity
        elif self.connect_f == 'IAND':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out


class Bottleneck(nn.Module):
    """
    实现ResNet网络的瓶颈架构
    """
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f=None):
        super(Bottleneck, self).__init__()
        
        self.connect_f = connect_f
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        
        width = int(planes * (base_width / 64.)) * groups
        
        self.layer1 = nn.Sequential(
            layer.Conv2d(inplanes, width, kernel_size=1, bias=False),
            norm_layer(width),
            neuron.IFNode(surrogate_function=surrogate.ATan()))
        
        self.layer2 = nn.Sequential(
            layer.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, groups = groups, dilation = dilation, bias=False),
            norm_layer(width),
            neuron.IFNode(surrogate_function=surrogate.ATan()))
        
        self.layer3 = nn.Sequential(
            layer.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False),
            norm_layer(planes * self.expansion),
            neuron.IFNode(surrogate_function=surrogate.ATan()))
   
        self.downsample = downsample
        
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        identity = x
        out = self.layer1(x)
        out = self.layer2(x)
        out = self.layer3(x)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        if self.connect_f == 'ADD':
            out += identity
        elif self.connect_f == 'AND':
            out *= identity
        elif self.connect_f == 'IAND':
            out = identity * (1. - out)
        else:
            raise NotImplementedError(self.connect_f)

        return out

# 对这两个架构的最后一层中的 BN 层进行设置，权重为0，且当连接方式为AND时，偏差置为1
def zero_init_blocks(net: nn.Module, connect_f: str):
    for m in net.modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.layer3.module[1].weight, 0)
            if connect_f == 'AND':
                nn.init.constant_(m.layer3.module[1].bias, 1)
        elif isinstance(m, BasicBlock):
            nn.init.constant_(m.layer2.module[1].weight, 0)
            if connect_f == 'AND':
                nn.init.constant_(m.layer2.module[1].bias, 1)

# 设计SEWResNet网络，原始的网络
class SEWResNet(nn.Module):

    def __init__(self, block, layers, input_channels = 3, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T=4, connect_f=None, drops = [0,0,0,0], p = [0.5,0.5,0.5,0.5]):
        super(SEWResNet, self).__init__()
        
        # 是否使用扩张率代替步长，一般这个操作应该用不上
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.T = T
        self.connect_f = connect_f
        self.groups = groups
        self.base_width = width_per_group
        self.drops = drops
        
        self.layer01 = nn.Sequential(
            layer.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes))
        
        self.layer02 = nn.Sequential(
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(kernel_size=3, stride=2, padding=1))
        functional.set_step_mode(self.layer02, step_mode='m')

        self.layer1 = self._make_layer(block, 64, layers[0], connect_f=connect_f)
        self.dp1 = layer.Dropout(p[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], connect_f=connect_f)
        self.dp2 = layer.Dropout(p[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], connect_f=connect_f)
        self.dp3 = layer.Dropout(p[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], connect_f=connect_f)
        self.dp4 = layer.Dropout(p[3])
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        # 最后一层的通道输出需要乘以扩张率
        self.fc = layer.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            zero_init_blocks(self, connect_f)
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, connect_f=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        # 使用扩张率代替步长
        if dilate:
            self.dilation *= stride
            stride = 1
        
        # 步长不等于1的时候，当然我们需要使用下采样层，当步长为1，但是在一个Block中，输入通道数不等于输出通道数的时候，
        # 我们仍然需要使用下采样层，因为我们每一个block块的输出一定是 planes * block.expansion，输入一定是self.inplanes
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    layer.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride = stride, bias=False),
                    norm_layer(planes * block.expansion),
                    neuron.IFNode(surrogate_function=surrogate.ATan()))
            functional.set_step_mode(downsample, step_mode='m')
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, connect_f))
        # 注意经过一个block之后，输入会发生通道的变化。
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, connect_f=connect_f))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.layer01(x)
        x.unsqueeze_(0) # 这一步经过实验可以去掉，因为x.repeat会自动添加维度
        x = x.repeat(self.T, 1, 1, 1, 1)  # T,B,C,H,W
        x = self.layer02(x)
        x = self.layer1(x)
        if self.drops[0]:
            x = self.dp1(x)
#             print("1")
        x = self.layer2(x)
        if self.drops[1]:
            x = self.dp2(x)
#             print("2")
        x = self.layer3(x)
        if self.drops[2]:
            x = self.dp3(x)
#             print("3")
        x = self.layer4(x)
        if self.drops[3]:
            x = self.dp4(x)
#             print("4")

        x = self.avgpool(x)
        x = torch.flatten(x, 2) # 将C,H,W合并为一体
        return self.fc(x.mean(dim=0)) # 首先在T这个维度求平均，接着使用Linear层

    def forward(self, x):
        return self._forward_impl(x)
    
def _sew_resnet(block, layers, **kwargs):
    model = SEWResNet(block, layers, **kwargs)
    return model


def sew_resnet18(**kwargs):
    return _sew_resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def sew_resnet34(**kwargs):
    return _sew_resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def sew_resnet50(**kwargs):
    return _sew_resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def sew_resnet101(**kwargs):
    return _sew_resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def sew_resnet152(**kwargs):
    return _sew_resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


# 设计SEWResNet网络的改版，区别主要在于是否在数据刚进入网络的时候就对数据进行T次复制
# 设计SEWResNet网络
class SEWResNet_2(nn.Module):

    def __init__(self, block, layers, input_channels = 3, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T=4, connect_f=None,dvs = '0'):
        super(SEWResNet_2, self).__init__()
        
        # 是否使用扩张率代替步长，一般这个操作应该用不上
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.T = T
        self.connect_f = connect_f
        self.groups = groups
        self.base_width = width_per_group
        self.dvs = dvs
        
        self.layer01 = nn.Sequential(
            layer.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes))
        
        self.layer02 = nn.Sequential(
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64, layers[0], connect_f=connect_f)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], connect_f=connect_f)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], connect_f=connect_f)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], connect_f=connect_f)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        # 最后一层的通道输出需要乘以扩张率
        self.fc = layer.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            zero_init_blocks(self, connect_f)
        
        functional.set_step_mode(self, step_mode='m')

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, connect_f=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        # 使用扩张率代替步长
        if dilate:
            self.dilation *= stride
            stride = 1
        
        # 步长不等于1的时候，当然我们需要使用下采样层，当步长为1，但是在一个Block中，输入通道数不等于输出通道数的时候，
        # 我们仍然需要使用下采样层，因为我们每一个block块的输出一定是 planes * block.expansion，输入一定是self.inplanes
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    layer.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride = stride, bias=False),
                    norm_layer(planes * block.expansion),
                    neuron.IFNode(surrogate_function=surrogate.ATan()))
#             functional.set_step_mode(downsample, step_mode='m')   
# 经过测试，如果需要整个网络为多步，只需要init函数尾部设置functional.set_step_mode(self, step_mode='m')
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, connect_f))
        # 注意经过一个block之后，输入会发生通道的变化。
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, connect_f=connect_f))

        return nn.Sequential(*layers)
    


    def _forward_impl(self, x):
        if self.dvs == '0':
            x.unsqueeze_(0) # 这一步经过实验可以去掉，因为x.repeat会自动添加维度
            x = x.repeat(self.T, 1, 1, 1, 1)  # T,B,C,H,W
        x = self.layer01(x)
        x = self.layer02(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2) # 将C,H,W合并为一体
        return self.fc(x.mean(dim=0)) # 首先在T这个维度求平均，接着使用Linear层

    def forward(self, x):
        return self._forward_impl(x)
    
def _sew_resnet_2(block, layers, **kwargs):
    model = SEWResNet_2(block, layers, **kwargs)
    return model


def sew_resnet18_2(**kwargs):
    return _sew_resnet_2(BasicBlock, [2, 2, 2, 2], **kwargs)


def sew_resnet34_2(**kwargs):
    return _sew_resnet_2(BasicBlock, [3, 4, 6, 3], **kwargs)


def sew_resnet50_2(**kwargs):
    return _sew_resnet_2(Bottleneck, [3, 4, 6, 3], **kwargs)


def sew_resnet101_2(**kwargs):
    return _sew_resnet_2(Bottleneck, [3, 4, 23, 3], **kwargs)


def sew_resnet152_2(**kwargs):
    return _sew_resnet_2(Bottleneck, [3, 8, 36, 3], **kwargs)


# spiking_resnet网络的实现
class SP_BasicBlock(nn.Module):
    """
    实现ResNet网络的第一个基础架构
    """
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f=None):
        super(SP_BasicBlock, self).__init__()
        
        self.connect_f = connect_f
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('SpikingBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SpikingBasicBlock")

        self.layer1 = nn.Sequential(
            layer.Conv2d(inplanes, planes, kernel_size=3, stride = stride, padding=1, bias=False),
            norm_layer(planes),
            neuron.IFNode(surrogate_function=surrogate.ATan()))
        
        self.layer2 = nn.Sequential(
            layer.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
            norm_layer(planes),
            )
        
        self.sn2 = neuron.IFNode(surrogate_function=surrogate.ATan())

        self.downsample = downsample
        
        # 设置所有的层为多步模式
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        identity = x
        out = self.layer2(self.layer1(x))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        
        return self.sn2(out)


class SP_Bottleneck(nn.Module):
    """
    实现ResNet网络的瓶颈架构
    """
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, connect_f=None):
        super(SP_Bottleneck, self).__init__()
        
        self.connect_f = connect_f
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        
        width = int(planes * (base_width / 64.)) * groups
        
        self.layer1 = nn.Sequential(
            layer.Conv2d(inplanes, width, kernel_size=1, bias=False),
            norm_layer(width),
            neuron.IFNode(surrogate_function=surrogate.ATan()))
        
        self.layer2 = nn.Sequential(
            layer.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, groups = groups, dilation = dilation, bias=False),
            norm_layer(width),
            neuron.IFNode(surrogate_function=surrogate.ATan()))
        
        self.layer3 = nn.Sequential(
            layer.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False),
            norm_layer(planes * self.expansion))
        
        self.sn3 = neuron.IFNode(surrogate_function=surrogate.ATan())
   
        self.downsample = downsample
        
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        identity = x
        out = self.layer3(self.layer2(self.layer1(x)))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity

        return self.sn3(out)

# 对这两个架构的最后一层中的 BN 层进行设置，权重为0，且当连接方式为AND时，偏差置为1
def SP_zero_init_blocks(net: nn.Module, connect_f: str):
    for m in net.modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.layer3.module[1].weight, 0)
        elif isinstance(m, BasicBlock):
            nn.init.constant_(m.layer2.module[1].weight, 0)

class SpikingResNet(nn.Module):

    def __init__(self, block, layers, input_channels = 3, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T=4, connect_f=None):
        super(SpikingResNet, self).__init__()
        
        # 是否使用扩张率代替步长，一般这个操作应该用不上
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.T = T
        self.connect_f = connect_f
        self.groups = groups
        self.base_width = width_per_group
        
        self.layer01 = nn.Sequential(
            layer.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes))
        
        self.layer02 = nn.Sequential(
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(kernel_size=3, stride=2, padding=1))
        functional.set_step_mode(self.layer02, step_mode='m')

        self.layer1 = self._make_layer(block, 64, layers[0], connect_f=connect_f)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], connect_f=connect_f)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], connect_f=connect_f)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], connect_f=connect_f)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        # 最后一层的通道输出需要乘以扩张率
        self.fc = layer.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            SP_zero_init_blocks(self, connect_f)
        
        

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, connect_f=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        # 使用扩张率代替步长
        if dilate:
            self.dilation *= stride
            stride = 1
        
        # 步长不等于1的时候，当然我们需要使用下采样层，当步长为1，但是在一个Block中，输入通道数不等于输出通道数的时候，
        # 我们仍然需要使用下采样层，因为我们每一个block块的输出一定是 planes * block.expansion，输入一定是self.inplanes
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    layer.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride = stride, bias=False),
                    norm_layer(planes * block.expansion))
            functional.set_step_mode(downsample, step_mode='m')
#             functional.set_step_mode(downsample, step_mode='m')   
# 经过测试，如果需要整个网络为多步，只需要init函数尾部设置functional.set_step_mode(self, step_mode='m')
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, connect_f))
        # 注意经过一个block之后，输入会发生通道的变化。
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, connect_f=connect_f))

        return nn.Sequential(*layers)
    


    def _forward_impl(self, x):
        x = self.layer01(x)
        # 将前面的层当成数据的变化，但是一般来说应该是先输入脉冲之后，在进行各种变化，所以从直觉上这种输入是否符合逻辑？
        x.unsqueeze_(0) # 这一步经过实验可以去掉，因为x.repeat会自动添加维度
        x = x.repeat(self.T, 1, 1, 1, 1)  # T,B,C,H,W
        x = self.layer02(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2) # 将C,H,W合并为一体
        return self.fc(x.mean(dim=0)) # 首先在T这个维度求平均，接着使用Linear层

    def forward(self, x):
        return self._forward_impl(x)
    
def _spiking_resnet(block, layers, **kwargs):
    model = SpikingResNet(block, layers, **kwargs)
    return model


def spiking_resnet18(**kwargs):
    return _spiking_resnet(SP_BasicBlock, [2, 2, 2, 2], **kwargs)


def spiking_resnet34(**kwargs):
    return _spiking_resnet(SP_BasicBlock, [3, 4, 6, 3], **kwargs)


def spiking_resnet50(**kwargs):
    return _spiking_resnet(SP_Bottleneck, [3, 4, 6, 3], **kwargs)


def spiking_resnet101(**kwargs):
    return _spiking_resnet(SP_Bottleneck, [3, 4, 23, 3], **kwargs)


def spiking_resnet152(**kwargs):
    return _spiking_resnet(SP_Bottleneck, [3, 8, 36, 3], **kwargs)



class SpikingResNet_2(nn.Module):

    def __init__(self, block, layers, input_channels = 3, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T=4, connect_f=None, dvs = "0"):
        super(SpikingResNet_2, self).__init__()
        
        # 是否使用扩张率代替步长，一般这个操作应该用不上
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.T = T
        self.connect_f = connect_f
        self.groups = groups
        self.base_width = width_per_group
        self.dvs = dvs
        
        self.layer01 = nn.Sequential(
            layer.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes))
        
        self.layer02 = nn.Sequential(
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(kernel_size=3, stride=2, padding=1))
        

        self.layer1 = self._make_layer(block, 64, layers[0], connect_f=connect_f)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], connect_f=connect_f)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], connect_f=connect_f)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], connect_f=connect_f)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        # 最后一层的通道输出需要乘以扩张率
        self.fc = layer.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            SP_zero_init_blocks(self, connect_f)
        
        functional.set_step_mode(self, step_mode='m')
        
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, connect_f=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        # 使用扩张率代替步长
        if dilate:
            self.dilation *= stride
            stride = 1
        
        # 步长不等于1的时候，当然我们需要使用下采样层，当步长为1，但是在一个Block中，输入通道数不等于输出通道数的时候，
        # 我们仍然需要使用下采样层，因为我们每一个block块的输出一定是 planes * block.expansion，输入一定是self.inplanes
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    layer.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride = stride, bias=False),
                    norm_layer(planes * block.expansion))
            functional.set_step_mode(downsample, step_mode='m')
#             functional.set_step_mode(downsample, step_mode='m')   
# 经过测试，如果需要整个网络为多步，只需要init函数尾部设置functional.set_step_mode(self, step_mode='m')
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, connect_f))
        # 注意经过一个block之后，输入会发生通道的变化。
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, connect_f=connect_f))

        return nn.Sequential(*layers)
    


    def _forward_impl(self, x):
        if self.dvs == '0':
            x.unsqueeze_(0) # 这一步经过实验可以去掉，因为x.repeat会自动添加维度
            x = x.repeat(self.T, 1, 1, 1, 1)  # T,B,C,H,W
        x = self.layer01(x)
        # 将前面的层当成数据的变化，但是一般来说应该是先输入脉冲之后，在进行各种变化，所以从直觉上这种输入是否符合逻辑？
        x = self.layer02(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2) # 将C,H,W合并为一体
        return self.fc(x.mean(dim=0)) # 首先在T这个维度求平均，接着使用Linear层

    def forward(self, x):
        return self._forward_impl(x)
    
def _spiking_resnet_2(block, layers, **kwargs):
    model = SpikingResNet_2(block, layers, **kwargs)
    return model


def spiking_resnet18_2(**kwargs):
    return _spiking_resnet_2(SP_BasicBlock, [2, 2, 2, 2], **kwargs)


def spiking_resnet34_2(**kwargs):
    return _spiking_resnet_2(SP_BasicBlock, [3, 4, 6, 3], **kwargs)


def spiking_resnet50_2(**kwargs):
    return _spiking_resnet_2(SP_Bottleneck, [3, 4, 6, 3], **kwargs)


def spiking_resnet101_2(**kwargs):
    return _spiking_resnet_2(SP_Bottleneck, [3, 4, 23, 3], **kwargs)


def spiking_resnet152_2(**kwargs):
    return _spiking_resnet(SP_Bottleneck, [3, 8, 36, 3], **kwargs)





class Re_BasicBlock(nn.Module):
    """
    基础块之1  两层卷积 第一个卷积的步长是可变的，第二个卷积步长固定为1
    使用BN时，Conv层的bias是不需要使用的，故设置为false
    downsample：为了能够在特征shape变化后还能使用残差连接，我们需要设置下采样卷积
    """
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(Re_BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # 下采样方法，当输入特征矩阵的长和宽与输出不一致时使用
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Re_Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Re_Bottleneck, self).__init__()
        
        # 使用resnetXt时，我们的width_per_group = 4 groups =32
        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 in_channel = 3,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(in_channel, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        # 若在块的开始  也就是从第二块开始 就必须进行这种所谓的下采样，否则就不需要下采样
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

    
def resnet18(in_channel = 3, num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(Re_BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channel = in_channel, include_top=include_top)

def resnet34(in_channel = 3, num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(Re_BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_channel = in_channel, include_top=include_top)


def resnet50(in_channel = 3, num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Re_Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_channel = in_channel, include_top=include_top)


def resnet101(in_channel = 3, num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Re_Bottleneck, [3, 4, 23, 3], num_classes=num_classes, in_channel = in_channel, include_top=include_top)


def resnext50_32x4d(in_channel = 3, num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Re_Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  in_channel = in_channel, 
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(in_channel = 3, num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Re_Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  in_channel = in_channel, 
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)



