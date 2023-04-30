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
            label: Optional[Dict[str, torch.Tensor]] = None,
            label_lengths: Optional[Dict[str, torch.Tensor]] = None,
            melody: Optional[Dict[str, torch.Tensor]] = None,
            melody_lengths: Optional[Dict[str, torch.Tensor]] = None,
            pitch: Optional[torch.Tensor] = None,
            pitch_lengths: Optional[torch.Tensor] = None,
            duration: Optional[Dict[str, torch.Tensor]] = None,
            duration_lengths: Optional[Dict[str, torch.Tensor]] = None,
            slur: torch.LongTensor = None,
            slur_lengths: torch.Tensor = None,
            spembs: Optional[torch.Tensor] = None,
            sids: Optional[torch.Tensor] = None,
            lids: Optional[torch.Tensor] = None,
            joint_training: bool = False,
            flag_IsValid=False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate outputs and return the loss tensor.

        Args:
            text (LongTensor): Batch of padded character ids (B, T_text).
            text_lengths (LongTensor): Batch of lengths of each input (B,).
            feats (Tensor): Batch of padded target features (B, T_feats, odim).
            feats_lengths (LongTensor): Batch of the lengths of each target (B,).
            label (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded label ids (B, Tmax).
            label_lengths (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of the lengths of padded label ids (B, ).
            melody (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of padded melody (B, Tmax).
            melody_lengths (Optional[Dict]): key is "lab" or "score";
                value (LongTensor): Batch of the lengths of padded melody (B, ).
            pitch (FloatTensor): Batch of padded f0 (B, Tmax).
            pitch_lengths (LongTensor): Batch of the lengths of padded f0 (B, ).
            duration (Optional[Dict]): key is "lab", "score_phn" or "score_syb";
                value (LongTensor): Batch of padded duration (B, Tmax).
            duration_length (Optional[Dict]): key is "lab", "score_phn" or "score_syb";
                value (LongTensor): Batch of the lengths of padded duration (B, ).
            slur (LongTensor): Batch of padded slur (B, Tmax).
            slur_lengths (LongTensor): Batch of the lengths of padded slur (B, ).
            spembs (Optional[Tensor]): Batch of speaker embeddings (B, spk_embed_dim).
            sids (Optional[Tensor]): Batch of speaker IDs (B, 1).
            lids (Optional[Tensor]): Batch of language IDs (B, 1).
            joint_training (bool): Whether to perform joint training with vocoder.

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value if not joint training else model outputs.
        """

        if joint_training:
            label = label
            midi = melody
            tempo = duration
            label_lengths = label_lengths
            midi_lengths = melody_lengths
            tempo_lengths = duration_lengths
            ds = duration
        else:
            label = label["score"]
            midi = melody["score"]
            tempo = duration["score_phn"]
            label_lengths = label_lengths["score"]
            midi_lengths = melody_lengths["score"]
            tempo_lengths = duration_lengths["score_phn"]
            ds = duration["lab"]

        text = text[:, : text_lengths.max()]  # for data-parallel
        feats = feats[:, : feats_lengths.max()]  # for data-parallel
        midi = midi[:, : midi_lengths.max()]  # for data-parallel
        label = label[:, : label_lengths.max()]  # for data-parallel
        tempo = tempo[:, : tempo_lengths.max()]  # for data-parallel
        batch_size = text.size(0)

        print("TEXT", text.shape)
        print("PITCH", pitch.shape)

        breakpoint()

        # x = torch.cat((text, pitch), dim=-1)
        # print("X", x.shape)

        # # Add eos at the last of sequence
        # xs = F.pad(text, [0, 1], "constant", self.padding_idx)
        # for i, l in enumerate(text_lengths):
        #     xs[i, l] = self.eos
        # ilens = text_lengths + 1

        raise NotImplementedError

    def inference(
            self,
            text: torch.Tensor,
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Return predicted output as a dict."""
        raise NotImplementedError
