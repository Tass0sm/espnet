# Copyright 2023 Tassos Manganaris
#  TODO: License

"""Singing-Tacotron related modules for ESPnet2."""

import logging
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.svs.abs_tts import AbsSVS

class SingingTacotron(AbsSVS):
    """
    TODO: Description

    .. _`Singing-Tacotron: Global duration control attention and dynamic filter
       for End-to-end singing voice synthesis`: https://arxiv.org/abs/2202.07907

    """

    def forward(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            feats: torch.Tensor,
            feats_lengths: torch.Tensor,
            **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate outputs and return the loss tensor."""
        raise NotImplementedError

    def inference(
            self,
            text: torch.Tensor,
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Return predicted output as a dict."""
        raise NotImplementedError
