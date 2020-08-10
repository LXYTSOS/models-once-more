# -*- coding: utf-8 -*-
"""
__author__ = liuxiangyu
__mtime__ = 2020/6/10 21:24
"""
import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __int__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            # (224+4-11)/4 + 1 = 55, 55 * 55 * 64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (55-3)/2 + 1 = 27, 27 * 27 * 64
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            # (27+4-5)/1 + 1 = 27, 27 * 27 * 192
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (27-3)/2 + 1 = 13, 13 * 13 * 192
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # (13+2-3)/1 + 1 = 13, 13 * 13 * 384
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # (13+2-3)/1 + 1 = 13, 13 * 13 * 256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # (13+2-3)/1 + 1 = 13, 13 * 13 * 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # (13-3)/2 + 1 = 6, 6 * 6 * 256
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6)) # input = (6, 6)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, progress=True, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = torch.load('path-to-pretrained-model')
        model.load_state_dict(state_dict)
    return model
