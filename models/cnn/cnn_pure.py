import torch.nn as nn
from .feature_extractor import CNNFeatureExtractor

class CNNPure(nn.Module):
    """
    CNN pura entrenada end-to-end.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = CNNFeatureExtractor()
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
