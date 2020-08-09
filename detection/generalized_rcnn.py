# -*- coding: utf-8 -*-
"""
__author__ = liuxiangyu
__mtime__ = 2020/8/7 22:01
"""

from collections import OrderedDict
import torch
from torch import nn


class GeneralizedRCNN(nn.Module):
    def __init__(self, backbone, rpn, roi_heads, transfrom):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transfrom
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_loasses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_loasses)
        losses.update(proposal_losses)

        if self.training:
            return losses

        return detections