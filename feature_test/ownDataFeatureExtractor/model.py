import torch.nn.functional as F
import torch.nn as nn
import torch
from torchvision.models import mobilenet_v2
import numpy as np


class FeatureExtractorNet(torch.nn.Module):
    def __init__(self, bottleneck_input_size=1280 * 11 * 4, use_classifier=False, num_features=32, num_classes=5):
        super(FeatureExtractorNet, self).__init__()
        mn = mobilenet_v2(pretrained=True)
        self.backbone = mn.features
        self.bottleneck = nn.Linear(bottleneck_input_size, num_features)
        self.classifier = nn.Linear(num_features, num_classes)
        self.use_classifier = use_classifier

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, num_flat_features(x))
        x = self.bottleneck(x)
        if self.use_classifier:
            x = self.classifier(x)
        return x


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

