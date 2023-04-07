"""Text-to-speech ESPnet model with asr component."""

from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.inversible_interface import InversibleInterface
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet2.tts.espnet_model import ESPnetTTSModel

# ASR
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr_transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):  # NOQA
        yield

class ESPnetUTTSModel(ESPnetTTSModel):
    """ESPnet model for text-to-speech task with asr component."""

    def __init__(
            self,
            feats_extract: Optional[AbsFeatsExtract],
            pitch_extract: Optional[AbsFeatsExtract],
            energy_extract: Optional[AbsFeatsExtract],
            normalize: Optional[AbsNormalize and InversibleInterface],
            pitch_normalize: Optional[AbsNormalize and InversibleInterface],
            energy_normalize: Optional[AbsNormalize and InversibleInterface],
            tts: AbsTTS,
            # ASR Components
            asr_model: ESPnetASRModel
    ):
        """Initialize ESPnetTTSModel module."""
        assert check_argument_types()
        super().__init__(
            feats_extract,
            pitch_extract,
            energy_extract,
            normalize,
            pitch_normalize,
            energy_normalize,
            tts
        )

        # Save parameters for asr encoder + forward
        self.ignore_id = asr_model.ignore_id
        self.is_encoder_whisper = asr_model.is_encoder_whisper
        self.ctc_weight = asr_model.ctc_weight
        self.interctc_weight = asr_model.interctc_weight

        # possibly don't register this as a submodule.
        self.m = {
            "asr_model": asr_model
        }

        self.asr_normalize = asr_model.normalize
        if self.asr_normalize is not None:
            self.asr_normalize.requires_grad_(False)

        self.asr_preencoder = asr_model.preencoder
        if self.asr_preencoder is not None:
            self.asr_preencoder.requires_grad_(False)

        self.asr_encoder = asr_model.encoder
        if self.asr_encoder is not None:
            self.asr_encoder.requires_grad_(False)

        self.asr_postencoder = asr_model.postencoder
        if self.asr_postencoder is not None:
            self.asr_postencoder.requires_grad_(False)

        self.asr_decoder = asr_model.decoder
        if self.asr_decoder is not None:
            self.asr_decoder.requires_grad_(False)

        self.asr_ctc = asr_model.ctc
        if self.asr_ctc is not None:
            self.asr_ctc.requires_grad_(False)


    # def asr_encode(
    #         self,
    #         feats: torch.Tensor,
    #         feats_lengths: torch.Tensor,
    #         **kwargs,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     with autocast(False):
    #         # Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
    #         if self.asr_normalize is not None:
    #             feats, feats_lengths = self.asr_normalize(feats, feats_lengths)

    #     # Pre-encoder, e.g. used for raw input data
    #     if self.asr_preencoder is not None:
    #         feats, feats_lengths = self.asr_preencoder(feats, feats_lengths)

    #     # 4. Forward encoder
    #     # feats: (Batch, Length, Dim)
    #     # -> encoder_out: (Batch, Length2, Dim2)
    #     if self.asr_encoder.interctc_use_conditioning:
    #         encoder_out, encoder_out_lens, _ = self.asr_encoder(
    #             feats, feats_lengths, ctc=self.ctc
    #         )
    #     else:
    #         encoder_out, encoder_out_lens, _ = self.asr_encoder(feats, feats_lengths)
    #     intermediate_outs = None
    #     if isinstance(encoder_out, tuple):
    #         intermediate_outs = encoder_out[1]
    #         encoder_out = encoder_out[0]

    #     # Post-encoder, e.g. NLU
    #     if self.asr_postencoder is not None:
    #         encoder_out, encoder_out_lens = self.asr_postencoder(
    #             encoder_out, encoder_out_lens
    #         )

    #     # the batch size of the encoder output is the same as the original batch
    #     # size.
    #     assert encoder_out.size(0) == feats.size(0), (
    #         encoder_out.size(),
    #         feats.size(0),
    #     )
    #     if (
    #         getattr(self.asr_encoder, "selfattention_layer_type", None) != "lf_selfattn"
    #         and not self.is_encoder_whisper
    #     ):
    #         assert encoder_out.size(-2) <= encoder_out_lens.max(), (
    #             encoder_out.size(),
    #             encoder_out_lens.max(),
    #         )

    #     if intermediate_outs is not None:
    #         return (encoder_out, intermediate_outs), encoder_out_lens

    #     return encoder_out, encoder_out_lens

    # def asr_forward(
    #     self,
    #     text: torch.Tensor,
    #     text_lengths: torch.Tensor,
    #     feats: torch.Tensor,
    #     feats_lengths: torch.Tensor,
    #     **kwargs,
    # ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    #     """Encoder + Decoder + Calc loss

    #     Args:
    #         text: (Batch, Length)
    #         text_lengths: (Batch,)
    #         feats: (Batch, Length, ...)
    #         feats_lengths: (Batch, )
    #         kwargs: "utt_id" is among the input.
    #     """
    #     assert text_lengths.dim() == 1, text_lengths.shape
    #     # Check that batch_size is unified
    #     assert (
    #         feats.shape[0]
    #         == feats_lengths.shape[0]
    #         == text.shape[0]
    #         == text_lengths.shape[0]
    #     ), (feats.shape, feats_lengths.shape, text.shape, text_lengths.shape)
    #     batch_size = feats.shape[0]

    #     text[text == -1] = self.ignore_id

    #     # for data-parallel
    #     text = text[:, : text_lengths.max()]

    #     # feats_denorm = self.normalize.inverse(
    #     #     feats.clone()[None]
    #     # )[0][0]

    #     # 1. Encoder
    #     encoder_out, encoder_out_lens = self.asr_encode(feats, feats_lengths)
    #     intermediate_outs = None
    #     if isinstance(encoder_out, tuple):
    #         intermediate_outs = encoder_out[1]
    #         encoder_out = encoder_out[0]

    #     loss_att, acc_att, cer_att, wer_att = None, None, None, None
    #     loss_ctc, cer_ctc = None, None
    #     loss_transducer, cer_transducer, wer_transducer = None, None, None
    #     stats = dict()

    #     # 1. CTC branch
    #     if self.ctc_weight != 0.0:
    #         loss_ctc, cer_ctc = self.m["asr_model"]._calc_ctc_loss(
    #             encoder_out, encoder_out_lens, text, text_lengths
    #         )

    #         # Collect CTC branch stats
    #         stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
    #         stats["cer_ctc"] = cer_ctc

    #     # Intermediate CTC (optional)
    #     loss_interctc = 0.0
    #     if self.interctc_weight != 0.0 and intermediate_outs is not None:
    #         for layer_idx, intermediate_out in intermediate_outs:
    #             # we assume intermediate_out has the same length & padding
    #             # as those of encoder_out

    #             # use auxillary ctc data if specified
    #             loss_ic = None
    #             if self.m["asr_model"].aux_ctc is not None:
    #                 idx_key = str(layer_idx)
    #                 if idx_key in self.m["asr_model"].aux_ctc:
    #                     aux_data_key = self.m["asr_model"].aux_ctc[idx_key]
    #                     aux_data_tensor = kwargs.get(aux_data_key, None)
    #                     aux_data_lengths = kwargs.get(aux_data_key + "_lengths", None)

    #                     if aux_data_tensor is not None and aux_data_lengths is not None:
    #                         loss_ic, cer_ic = self.m["asr_model"]._calc_ctc_loss(
    #                             intermediate_out,
    #                             encoder_out_lens,
    #                             aux_data_tensor,
    #                             aux_data_lengths,
    #                         )
    #                     else:
    #                         raise Exception(
    #                             "Aux. CTC tasks were specified but no data was found"
    #                         )
    #             if loss_ic is None:
    #                 loss_ic, cer_ic = self.m["asr_model"]._calc_ctc_loss(
    #                     intermediate_out, encoder_out_lens, text, text_lengths
    #                 )
    #             loss_interctc = loss_interctc + loss_ic

    #             # Collect Intermedaite CTC stats
    #             stats["loss_interctc_layer{}".format(layer_idx)] = (
    #                 loss_ic.detach() if loss_ic is not None else None
    #             )
    #             stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic

    #         loss_interctc = loss_interctc / len(intermediate_outs)

    #         # calculate whole encoder loss
    #         loss_ctc = (
    #             1 - self.interctc_weight
    #         ) * loss_ctc + self.interctc_weight * loss_interctc


    #     if self.m["asr_model"].use_transducer_decoder:
    #         # 2a. Transducer decoder branch
    #         (
    #             loss_transducer,
    #             cer_transducer,
    #             wer_transducer,
    #         ) = self.m["asr_model"]._calc_transducer_loss(
    #             encoder_out,
    #             encoder_out_lens,
    #             text,
    #         )

    #         if loss_ctc is not None:
    #             loss = loss_transducer + (self.ctc_weight * loss_ctc)
    #         else:
    #             loss = loss_transducer

    #         # Collect Transducer branch stats
    #         stats["loss_transducer"] = (
    #             loss_transducer.detach() if loss_transducer is not None else None
    #         )
    #         stats["cer_transducer"] = cer_transducer
    #         stats["wer_transducer"] = wer_transducer

    #     else:
    #         # 2b. Attention decoder branch
    #         if self.ctc_weight != 1.0:
    #             loss_att, acc_att, cer_att, wer_att = self.m["asr_model"]._calc_att_loss(
    #                 encoder_out, encoder_out_lens, text, text_lengths
    #             )

    #         # 3. CTC-Att loss definition
    #         if self.ctc_weight == 0.0:
    #             loss = loss_att
    #         elif self.ctc_weight == 1.0:
    #             loss = loss_ctc
    #         else:
    #             loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

    #         # Collect Attn branch stats
    #         stats["loss_att"] = loss_att.detach() if loss_att is not None else None
    #         stats["acc"] = acc_att
    #         stats["cer"] = cer_att
    #         stats["wer"] = wer_att

    #     # Collect total loss stats
    #     stats["loss"] = loss.detach()

    #     # force_gatherable: to-device and to-tensor if scalar for DataParallel
    #     loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
    #     return loss, stats, weight

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        durations_lengths: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        pitch_lengths: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        energy_lengths: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        lambda_text2mel: float = 1.0,
        lambda_asr: float = 0.08,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Caclualte outputs and return the loss tensor.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            speech (Tensor): Speech waveform tensor (B, T_wav).
            speech_lengths (Tensor): Speech length tensor (B,).
            duration (Optional[Tensor]): Duration tensor.
            duration_lengths (Optional[Tensor]): Duration length tensor (B,).
            pitch (Optional[Tensor]): Pitch tensor.
            pitch_lengths (Optional[Tensor]): Pitch length tensor (B,).
            energy (Optional[Tensor]): Energy tensor.
            energy_lengths (Optional[Tensor]): Energy length tensor (B,).
            spembs (Optional[Tensor]): Speaker embedding tensor (B, D).
            sids (Optional[Tensor]): Speaker ID tensor (B, 1).
            lids (Optional[Tensor]): Language ID tensor (B, 1).
            kwargs: "utt_id" is among the input.

        Returns:
            Tensor: Loss scalar tensor.
            Dict[str, float]: Statistics to be monitored.
            Tensor: Weight tensor to summarize losses.

        """
        with autocast(False):
            # Extract features
            if self.feats_extract is not None:
                feats, feats_lengths = self.feats_extract(speech, speech_lengths)
            else:
                # Use precalculated feats (feats_type != raw case)
                feats, feats_lengths = speech, speech_lengths

            # Extract auxiliary features
            if self.pitch_extract is not None and pitch is None:
                pitch, pitch_lengths = self.pitch_extract(
                    speech,
                    speech_lengths,
                    feats_lengths=feats_lengths,
                    durations=durations,
                    durations_lengths=durations_lengths,
                )
            if self.energy_extract is not None and energy is None:
                energy, energy_lengths = self.energy_extract(
                    speech,
                    speech_lengths,
                    feats_lengths=feats_lengths,
                    durations=durations,
                    durations_lengths=durations_lengths,
                )

            # Normalize
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)
            if self.pitch_normalize is not None:
                pitch, pitch_lengths = self.pitch_normalize(pitch, pitch_lengths)
            if self.energy_normalize is not None:
                energy, energy_lengths = self.energy_normalize(energy, energy_lengths)

        # Make batch for tts inputs
        batch = dict(
            text=text,
            text_lengths=text_lengths,
            feats=feats,
            feats_lengths=feats_lengths,
        )

        # Update batch for additional auxiliary inputs
        if spembs is not None:
            batch.update(spembs=spembs)
        if sids is not None:
            batch.update(sids=sids)
        if lids is not None:
            batch.update(lids=lids)
        if durations is not None:
            batch.update(durations=durations, durations_lengths=durations_lengths)
        if self.pitch_extract is not None and pitch is not None:
            batch.update(pitch=pitch, pitch_lengths=pitch_lengths)
        if self.energy_extract is not None and energy is not None:
            batch.update(energy=energy, energy_lengths=energy_lengths)
        if self.tts.require_raw_speech:
            batch.update(speech=speech, speech_lengths=speech_lengths)

        text2mel_loss, stats, feats_gen = self.tts(**batch, utts_training=True)
        text2mel_loss = lambda_text2mel * text2mel_loss
        stats.update(text2mel_loss=text2mel_loss.item())

        # asr_loss, asr_stats, asr_weight = self.asr_forward(
        #     text, text_lengths, feats_gen, feats_lengths
        # )
        # asr_loss = lambda_asr * asr_loss
        # stats.update(asr_loss=asr_loss.item())
        # stats.update(**asr_stats);

        # this is okay. this computation results in the loss leaf tensor of the
        # computation graph.  one edge points directly to the tts model. another
        # edge points to the asr model graph, which isn't storing any
        # gradients. but the backpropagation still moves through it.
        loss = text2mel_loss
        # + asr_loss

        stats.update(loss=loss.item())

        batch_size = text.size(0)
        loss, stats, weight = force_gatherable(
            (loss, stats, batch_size), loss.device
        )

        return loss, stats, weight
