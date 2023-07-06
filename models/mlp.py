import torch
from torch import nn

def linear_block(
    in_channels: int,
    out_channels: int,
    bias: bool = True,
    activation=None,
    normalization=nn.Identity()
) -> nn.Module:
    bias = not normalization and bias
    linear_module = nn.Linear(
        in_features=in_channels,
        out_features=out_channels,
        bias=bias,
    )
    return nn.Sequential(linear_module, normalization, activation)