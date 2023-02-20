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
    AxialAttentionWithoutPosition,
    AxialAttentionWithPosition,
    AxialAttentionWithPositionAndGate
)

class AminEncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        width (int): Input width dimension.
        height (int): Input height dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """

    def __init__(
        self,
        height,
        width,
        channels,
        hidden_channels,
        self_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
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

        self.self_attn = self_attn
        self.flatten = nn.Conv2d(hidden_channels, channels, 1)
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm((height, width))
        self.norm2 = LayerNorm((height, width))
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

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

        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            return x, mask

        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.height, self.width)
            x_q = x[:, -1:, :, :]
            residual = residual[:, -1:, :, :]
            mask = None if mask is None else mask[:, -1:, :, :]

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            x = residual + stoch_layer_coeff * self.dropout(
                self.self_attn(x_q, x, x, mask)
            )

        if not self.normalize_before:
            x = self.norm1(x)


        # compress down to a single matrix for the feed forward step. Is this
        # like WO in a normal transformer?
        b, t, c, h, w = x.shape
        x = x.contiguous().view(b * t, c, h, w)
        x = self.flatten(x)
        x = x.contiguous().view(b, t, self.channels, h, w)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))

        if not self.normalize_before:
            x = self.norm2(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, mask
