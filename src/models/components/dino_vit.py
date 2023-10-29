
import torch.nn as nn


class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self, net: nn.Module, intermediate:int = 256, num_classes: int = 4):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = net
        self.transformer.head = nn.Sequential(
            nn.Linear(384, intermediate),
            nn.ReLU(),
            nn.Linear(intermediate, num_classes)
        )

    def forward(self, x):
        return self.transformer(x)
