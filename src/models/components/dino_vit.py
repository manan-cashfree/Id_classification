import torch.nn as nn


class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self, net: nn.Module, in_features: int = 768, num_classes: int = 4, layer_params: list = []):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = net
        self.in_features = in_features
        self.num_classes = num_classes
        self.layer_params = layer_params
        self.transformer.head = self.create_mlp_head()

    def create_mlp_head(self):
        num_layers = len(self.layer_params)
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(self.in_features, self.layer_params[i]))
            else:
                layers.append(nn.Linear(self.layer_params[i - 1], self.layer_params[i]))
            layers.append(nn.ReLU())
        if num_layers == 0:
            layers.append(nn.Linear(self.in_features, self.num_classes))
        else:
            layers.append(nn.Linear(self.layer_params[-1], self.num_classes))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.transformer(x)
