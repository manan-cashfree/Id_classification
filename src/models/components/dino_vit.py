# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm


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
            layers.append(nn.GELU())
        if num_layers == 0:
            layers.append(nn.Linear(self.in_features, self.num_classes))
        else:
            layers.append(nn.Linear(self.layer_params[-1], self.num_classes))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.transformer(x)


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias)
        self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)