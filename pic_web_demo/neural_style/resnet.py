from collections import namedtuple

import torch
from torchvision import models


class ResNet18(torch.nn.Module):
    """
    定义ResNet18网络
    使用预训练模型 提取相应层的输出结果
    """
    
    def __init__(self, requires_grad=False):
        super(ResNet18, self).__init__()
        # 加载预训练模型
        resnet_pretrained_features = models.resnet18(pretrained=True)
        # 定义不同的模块 模块初始化
        self.slice0 = torch.nn.Sequential()
        self.slice0.add_module('1', resnet_pretrained_features.conv1)
        self.slice0.add_module('2', resnet_pretrained_features.bn1)
        self.slice0.add_module('3', resnet_pretrained_features.relu)
        self.slice0.add_module('4', resnet_pretrained_features.maxpool)
        self.slice1 = resnet_pretrained_features.layer1
        self.slice2 = resnet_pretrained_features.layer2
        self.slice3 = resnet_pretrained_features.layer3
        self.slice4 = resnet_pretrained_features.layer4
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice0(X)
        h = self.slice1(h)
        layer1 = h
        h = self.slice2(h)
        layer2 = h
        h = self.slice3(h)
        layer3 = h
        h = self.slice4(h)
        layer4 = h
        # 返回不同模块的输出结果
        resnet_outputs = namedtuple("ResNetOutputs", ['layer1', 'layer2', 'layer3', 'layer4'])
        out = resnet_outputs(layer1, layer2, layer3, layer4)
        return out
