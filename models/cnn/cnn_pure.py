import torch.nn as nn
from torch.nn.functional import dropout

from .feature_extractor import CNNFeatureExtractor

class CNNPure(nn.Module):
    """
    CNN pura entrenada end-to-end, con regularización Dropout.
    """
    def __init__(self, num_classes=2, dropout_p=0.3):
        super().__init__()
        self.features = CNNFeatureExtractor()
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
