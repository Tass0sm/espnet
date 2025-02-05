# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer encoder definition."""

from typing import List, Optional, Tuple

import math
import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.amin_encoder_layer import AminEncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling1,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)


class AminTransformerEncoder(AbsEncoder):
    """Transformer encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        positionwise_layer_type: linear of conv1d
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
        padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        input_size: int,
        output_width: int = 8,
        output_height: int = 64,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: Optional[str] = "linear",
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        attention_type: str = "without-position",
        input_channels: int = 1,
        hidden_channels: int = 2,
        groups: int = 1
    ):
        assert check_argument_types()
        super().__init__()

        # after embedded to the output height, chunk it according to
        # output_width. The patches all have output_width * output_height size.
        self.output_width = output_width
        self.output_height = output_height
        output_size = output_height * output_width
        self._output_size = output_size

        if input_layer == "linear":
            # use this to go down from 80 to 64 freqs
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_height),
                torch.nn.LayerNorm(output_height),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(output_height, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(input_size, output_height, dropout_rate)
        elif input_layer == "conv2d1":
            self.embed = Conv2dSubsampling1(input_size, output_height, dropout_rate)
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(input_size, output_height, dropout_rate)
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(input_size, output_height, dropout_rate)
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(input_size, output_height, dropout_rate)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_height, padding_idx=padding_idx),
                pos_enc_class(output_height, positional_dropout_rate),
            )
        elif input_layer is None:
            if input_size == output_height:
                self.embed = None
            else:
                self.embed = torch.nn.Linear(input_size, output_height)
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.normalize_before = normalize_before

        if positionwise_layer_type == "linear":
            positionwise_layer = torch.nn.Sequential(
                torch.nn.Flatten(start_dim=2),
                PositionwiseFeedForward(output_size, linear_units, dropout_rate),
                torch.nn.Unflatten(2, torch.Size([input_channels, output_height, output_width]))
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        self.encoders = repeat(
            num_blocks,
            lambda lnum: AminEncoderLayer(
                output_height,
                output_width,
                input_channels,
                hidden_channels,
                groups,
                attention_type,
                positionwise_layer,
                dropout_rate,
                normalize_before,
            ),
        )

        if self.normalize_before:
            self.after_norm = LayerNorm((output_height, output_width))

        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 < min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc: CTC = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if self.embed is None:
            xs_pad = xs_pad
        elif (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling1)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)
            if masks is not None:
                masks = masks[:, :, ::self.output_width]

        b, t, f = xs_pad.shape
        new_t = math.ceil(t / self.output_width) * self.output_width
        padding_amounts = (0, 0, 0, new_t - t, 0, 0)
        xs_pad = F.pad(xs_pad, padding_amounts, "constant", 0)

        b, t, f = xs_pad.shape
        xs_pad = xs_pad.reshape(b, t // self.output_width, 1, self.output_height, self.output_width)
        # (Batches, Time, Channels, Height, Width)

        intermediate_outs = []
        if len(self.interctc_layer_idx) == 0:
            xs_pad, masks = self.encoders(xs_pad, masks)
        else:
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs_pad, masks = encoder_layer(xs_pad, masks)

                if layer_idx + 1 in self.interctc_layer_idx:
                    encoder_out = xs_pad

                    # intermediate outputs are also normalized
                    if self.normalize_before:
                        encoder_out = self.after_norm(encoder_out)

                    intermediate_outs.append((layer_idx + 1, encoder_out))

                    if self.interctc_use_conditioning:
                        ctc_out = ctc.softmax(encoder_out)
                        xs_pad = xs_pad + self.conditioning_layer(ctc_out)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        b, t, c, h, w = xs_pad.shape
        xs_pad = xs_pad.contiguous().view(b, t, c * h * w)

        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens, None
