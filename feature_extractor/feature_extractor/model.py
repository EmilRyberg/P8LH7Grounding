import torch.nn.functional as F
import torch.nn as nn
import torch
from torchvision.models import mobilenet_v2
import numpy as np


class FeatureExtractorNet(torch.nn.Module):
    def __init__(self, bottleneck_input_size=1280, use_classifier=False, num_features=32, num_classes=5):
        super(FeatureExtractorNet, self).__init__()
        mn = mobilenet_v2(pretrained=True)
        self.backbone = mn.features
        self.bottleneck = nn.Linear(bottleneck_input_size, num_features)
        self.classifier = nn.Linear(num_features, num_classes)
        self.use_classifier = use_classifier
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.bottleneck(x)
        if self.use_classifier:
            x = self.classifier(x)
        return x

