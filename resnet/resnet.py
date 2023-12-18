import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResidualBlock1(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        intermediate_channels: int,
        identity_downsample :Callable,
        stride: int = 1,
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        
        self.conv1 = conv3x3(in_channels, intermediate_channels, stride)
        self.bn1 = norm_layer(intermediate_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(intermediate_channels, intermediate_channels)
        self.bn2 = norm_layer(intermediate_channels)
        self.stride = stride
        self.identity_downsample = identity_downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x.clone()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)



        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        out += identity
        out = self.relu(out)
        return out
    

class ResidualBlock2(nn.Module):
def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1,expansion = 4) ->Tensor:
    super().__init__()
    self.expansion = expansion
    self.conv1 = nn.Conv2d(
        in_channels,
        intermediate_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False,
    )
    self.bn1 = nn.BatchNorm2d(intermediate_channels)
    self.conv2 = nn.Conv2d(
        intermediate_channels,
        intermediate_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )
    self.bn2 = nn.BatchNorm2d(intermediate_channels)
    self.conv3 = nn.Conv2d(
        intermediate_channels,
        intermediate_channels * self.expansion,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False,
    )
    self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
    self.relu = nn.ReLU()
    self.identity_downsample = identity_downsample
    self.stride = stride

def forward(self, x) -> Tensor:
    identity = x.clone()
    
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.bn3(x)
    if self.identity_downsample is not None:
        identity = self.identity_downsample(identity)

    x += identity
    x = self.relu(x)
    return x


class Resnet(nn.Module):
    def __init__(self,block:Callable,layers: list,image_channels,expansion:int = 1,num_classes: int = 1000) -> None:
        super(Resnet,self).__init__()

        self.expansion = expansion
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )

        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )

        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)

    
    def _make_layer(self,block,num_residual_blocks,intermediate_channels,stride =1) -> nn.Sequential:

        layers = []
        
        if self.expansion == 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )
            layers.append(
                block(self.in_channels,intermediate_channels,identity_downsample,stride)
            )      
            self.in_channels = intermediate_channels * 4                                                                              
            for i in range(num_residual_blocks-1):
                layers.append(
                    block(self.in_channels,intermediate_channels)
                )
                
        elif self.expansion ==1:
            

            if intermediate_channels ==64:
                layers.append(
                    block(intermediate_channels,intermediate_channels,None,stride)
                )
            else:
                identity_downsample = nn.Sequential(
                                                    nn.Conv2d(
                                                        intermediate_channels//2,
                                                        intermediate_channels,
                                                        kernel_size=1,
                                                        stride=2,
                                                        padding =0,

                                                        bias=False,
                                                    ),
                                                    nn.BatchNorm2d(intermediate_channels),
                                                )
                layers.append(
                    block(intermediate_channels//2,intermediate_channels,identity_downsample,2)
                )
                
            for i in range(num_residual_blocks-1):
                layers.append(
                    block(intermediate_channels,intermediate_channels,None)
                )

        return nn.Sequential(*layers)
    

    def forward(self,x) -> Tensor:

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        print(x.size())
        x = self.fc(x)
        return x 
       

def ResNet18(num_classes=1000):
    return Resnet(ResidualBlock1,[2,2,2,2],3,1,num_classes)

def ResNet34(num_classes=1000):
    return Resnet(ResidualBlock1,[3,4,6,3],3,1,num_classes)

def ResNet50(num_classes=1000):
    return Resnet(ResidualBlock2,[3,4,6,3],3,4,num_classes)

def ResNet101(num_classes=1000):
    return Resnet(ResidualBlock2,[3,4,23,3],3,4,num_classes)
def ResNet152(num_classes=1000):
    return Resnet(ResidualBlock2,[3,8,36,3],3,4,num_classes)