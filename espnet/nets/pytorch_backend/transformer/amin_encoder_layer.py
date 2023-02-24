#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder self-attention layer definition."""

import torch
from torch import nn

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm

from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,
    AxialAttention
)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class AminEncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        width (int): Input width dimension.
        height (int): Input height dimension.
        channels (int): channels of tensor coming into and out of the encoder layer.
        hidden_channels (int): channels of tensor coming out of attention layer.
        attention_type (str): type of attention (without position, with position, with position and gate)
    """

    def __init__(
        self,
        height,
        width,
        channels,
        hidden_channels,
        groups,
        attention_type,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super(AminEncoderLayer, self).__init__()
        size = height * width
        self.height = height
        self.width = width
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.size = size

        self.conv_down = conv1x1(channels, hidden_channels)
        self.bn1 = LayerNorm((height, width))

        if attention_type == "without-position":
            self.height_block = AxialAttention(hidden_channels, hidden_channels, height, groups=groups, width=False)
            self.width_block = AxialAttention(hidden_channels, hidden_channels, width, groups=groups, width=True)
        elif attention_type == "with-position":
            self.height_block = AxialAttention(hidden_channels, hidden_channels, height, groups=groups, width=False, with_position=True)
            self.width_block = AxialAttention(hidden_channels, hidden_channels, width, groups=groups, width=True, with_position=True)
        elif attention_type == "with-position-gated":
            self.height_block = AxialAttention(hidden_channels, hidden_channels, height, groups=groups, width=False, with_position=True, with_gate=True)
            self.width_block = AxialAttention(hidden_channels, hidden_channels, width, groups=groups, width=True, with_position=True, with_gate=True)
        else:
            raise NotImplementedError("Bad attention type")

        self.bn2 = LayerNorm((height, width))
        self.conv_up = conv1x1(hidden_channels, channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, mask, cache=None):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, channels, height, width).
            mask (torch.Tensor): Mask tensor for the input (#batch, 1, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, height, width).

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels, height, width).
            torch.Tensor: Mask tensor (#batch, 1, time).

        """

        b, t, c, h, w = x.shape
        x = x.contiguous().view(b * t, c, h, w)

        identity = x

        out = self.conv_down(x) # expand c dim to hidden_channels
        out = self.bn1(out)
        out = self.relu(out)
        out = self.height_block(None, out, None, mask)
        out = self.width_block(None, out, None, mask)
        out = self.relu(out)
        out = self.conv_up(out) # contract back to normal number of channels
        out = self.bn2(x)
        out += identity
        out = self.relu(out)

        x = x.contiguous().view(b, t, c, h, w)

        return x, mask
