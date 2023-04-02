# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer-TTS related modules."""

from typing import Dict, Optional, Sequence, Tuple, Union
from pathlib import Path

import soundfile
import Levenshtein
import logging

import numpy as np
import torch
import torchtext
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import parse as V
from typeguard import check_argument_types


from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.inversible_interface import InversibleInterface
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.torch_utils.initialize import initialize
from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.utils import ParallelWaveGANPretrainedVocoder
from espnet2.tts.gst.style_encoder import StyleEncoder
from espnet.nets.pytorch_backend.e2e_tts_transformer import (
    GuidedMultiHeadAttentionLoss,
    TransformerLoss,
)
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask, make_pad_mask
from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet
from espnet.nets.pytorch_backend.tacotron2.decoder import Prenet as DecoderPrenet
from espnet.nets.pytorch_backend.tacotron2.encoder import Encoder as EncoderPrenet
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask

from espnet2.utts.tts.abs_utts_tts import AbsUTTSTTS
from espnet2.tts.transformer import Transformer

class Transformer(Transformer, AbsUTTSTTS):

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate forward propagation.

        Args:
            text (LongTensor): Batch of padded character ids (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input batch (B,).

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value if not joint training else model outputs.

        """
        text = text[:, : text_lengths.max()]  # for data-parallel
        batch_size = text.size(0)

        # Add eos at the last of sequence
        xs = F.pad(text, [0, 1], "constant", self.padding_idx)
        for i, l in enumerate(text_lengths):
            xs[i, l] = self.eos
        ilens = text_lengths + 1

        # ys = feats
        # olens = feats_lengths

        # # make labels for stop prediction
        # labels = make_pad_mask(olens - 1).to(ys.device, ys.dtype)
        # labels = F.pad(labels, [0, 1], "constant", 1.0)

        # calculate transformer outputs
        after_outs, before_outs, logits = self._forward(
            xs=xs,
            ilens=ilens,
        )

        # # modifiy mod part of groundtruth
        # olens_in = olens
        # if self.reduction_factor > 1:
        #     assert olens.ge(
        #         self.reduction_factor
        #     ).all(), "Output length must be greater than or equal to reduction factor."
        #     olens_in = olens.new([olen // self.reduction_factor for olen in olens])
        #     olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
        #     max_olen = max(olens)
        #     ys = ys[:, :max_olen]
        #     labels = labels[:, :max_olen]
        #     labels = torch.scatter(
        #         labels, 1, (olens - 1).unsqueeze(1), 1.0
        #     )  # see #3388

        # # calculate loss values
        # l1_loss, l2_loss, bce_loss, asr_loss = self.criterion(
        #     after_outs, before_outs, logits, ys, labels, olens, encoded, encoded_lengths
        # )
        # if self.loss_type == "L1":
        #     loss = l1_loss + bce_loss
        # elif self.loss_type == "L2":
        #     loss = l2_loss + bce_loss
        # elif self.loss_type == "L1+L2":
        #     loss = l1_loss + l2_loss + bce_loss
        # elif self.loss_type == "L1+L2+ASR":
        #     loss = l1_loss + l2_loss + bce_loss + asr_loss
        # elif self.loss_type == "ASR":
        #     loss = asr_loss
        # else:
        #     raise ValueError("unknown --loss-type " + self.loss_type)

        # stats = dict(
        #     l1_loss=l1_loss.item(),
        #     l2_loss=l2_loss.item(),
        #     bce_loss=bce_loss.item(),
        #     asr_loss=asr_loss.item()
        # )

        # # calculate guided attention loss
        # if self.use_guided_attn_loss:
        #     # calculate for encoder
        #     if "encoder" in self.modules_applied_guided_attn:
        #         att_ws = []
        #         for idx, layer_idx in enumerate(
        #             reversed(range(len(self.encoder.encoders)))
        #         ):
        #             att_ws += [
        #                 self.encoder.encoders[layer_idx].self_attn.attn[
        #                     :, : self.num_heads_applied_guided_attn
        #                 ]
        #             ]
        #             if idx + 1 == self.num_layers_applied_guided_attn:
        #                 break
        #         att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_text, T_text)
        #         enc_attn_loss = self.attn_criterion(att_ws, ilens, ilens)
        #         loss = loss + enc_attn_loss
        #         stats.update(enc_attn_loss=enc_attn_loss.item())
        #     # calculate for decoder
        #     if "decoder" in self.modules_applied_guided_attn:
        #         att_ws = []
        #         for idx, layer_idx in enumerate(
        #             reversed(range(len(self.decoder.decoders)))
        #         ):
        #             att_ws += [
        #                 self.decoder.decoders[layer_idx].self_attn.attn[
        #                     :, : self.num_heads_applied_guided_attn
        #                 ]
        #             ]
        #             if idx + 1 == self.num_layers_applied_guided_attn:
        #                 break
        #         att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_feats, T_feats)
        #         dec_attn_loss = self.attn_criterion(att_ws, olens_in, olens_in)
        #         loss = loss + dec_attn_loss
        #         stats.update(dec_attn_loss=dec_attn_loss.item())
        #     # calculate for encoder-decoder
        #     if "encoder-decoder" in self.modules_applied_guided_attn:
        #         att_ws = []
        #         for idx, layer_idx in enumerate(
        #             reversed(range(len(self.decoder.decoders)))
        #         ):
        #             att_ws += [
        #                 self.decoder.decoders[layer_idx].src_attn.attn[
        #                     :, : self.num_heads_applied_guided_attn
        #                 ]
        #             ]
        #             if idx + 1 == self.num_layers_applied_guided_attn:
        #                 break
        #         att_ws = torch.cat(att_ws, dim=1)  # (B, H*L, T_feats, T_text)
        #         enc_dec_attn_loss = self.attn_criterion(att_ws, ilens, olens_in)
        #         loss = loss + enc_dec_attn_loss
        #         stats.update(enc_dec_attn_loss=enc_dec_attn_loss.item())

        # # report extra information
        # if self.use_scaled_pos_enc:
        #     stats.update(
        #         encoder_alpha=self.encoder.embed[-1].alpha.data.item(),
        #         decoder_alpha=self.decoder.embed[-1].alpha.data.item(),
        #     )

        # if not joint_training:
        #     stats.update(loss=loss.item())
        #     loss, stats, weight = force_gatherable(
        #         (loss, stats, batch_size), loss.device
        #     )
        #     return loss, stats, weight
        # else:
        #     return loss, stats, after_outs

    def _forward(
        self,
        xs: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # forward encoder
        x_masks = self._source_mask(ilens)
        hs, h_masks = self.encoder(xs, x_masks)

        # integrate with GST
        if self.use_gst:
            style_embs = self.gst(ys)
            hs = hs + style_embs.unsqueeze(1)

        # integrate with SID and LID embeddings
        if self.spks is not None:
            sid_embs = self.sid_emb(sids.view(-1))
            hs = hs + sid_embs.unsqueeze(1)
        if self.langs is not None:
            lid_embs = self.lid_emb(lids.view(-1))
            hs = hs + lid_embs.unsqueeze(1)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)

        # thin out frames for reduction factor
        # (B, T_feats, odim) ->  (B, T_feats//r, odim)
        if self.reduction_factor > 1:
            ys_in = ys[:, self.reduction_factor - 1 :: self.reduction_factor]
            olens_in = olens.new([olen // self.reduction_factor for olen in olens])
        else:
            ys_in, olens_in = ys, olens

        # add first zero frame and remove last frame for auto-regressive
        ys_in = self._add_first_frame_and_remove_last_frame(ys_in)

        # forward decoder
        y_masks = self._target_mask(olens_in)
        zs, _ = self.decoder(ys_in, y_masks, hs, h_masks)
        # (B, T_feats//r, odim * r) -> (B, T_feats//r * r, odim)
        before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)
        # (B, T_feats//r, r) -> (B, T_feats//r * r)
        logits = self.prob_out(zs).view(zs.size(0), -1)

        # postnet -> (B, T_feats//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        return after_outs, before_outs, logits
