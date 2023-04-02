""""Unsupervised (informed by an ASR model) Text-to-speech ESPnet model."""

from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import torch
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.inversible_interface import InversibleInterface
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract

from espnet2.utts.tts.abs_utts_tts import AbsUTTSTTS

# ASR
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
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (  # noqa: H301
    LabelSmoothingLoss,
)



if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):  # NOQA
        yield


class ESPnetUTTSModel(AbsESPnetModel):
    """ESPnet model for the unsupervised text-to-speech task."""

    def __init__(
            self,
            # TTS Side (tokens -> something)
            tts_module: AbsUTTSTTS,
            # ASR Side (something -> tokens')
            asr_normalize: Optional[AbsNormalize and InversibleInterface],
            asr_preencoder: Optional[AbsPreEncoder],
            asr_encoder: AbsEncoder,
            asr_postencoder: Optional[AbsPostEncoder],
            asr_decoder: Optional[AbsDecoder],
            ctc: CTC,
    ):
        """Initialize ESPnetTTSModel module."""
        assert check_argument_types()
        super().__init__()
        self.asr_normalize = asr_normalize
        self.tts_module = tts_module

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Caclualte outputs and return the loss tensor.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).
            kwargs: anything else...

        Returns:
            Tensor: Loss scalar tensor.
            Dict[str, float]: Statistics to be monitored.
            Tensor: Weight tensor to summarize losses.

        """
        # Make batch for utts inputs
        batch = dict(
            text=text,
            text_lengths=text_lengths,
        )

        print("TTS", type(self.tts_module))

        feats, feats_lengths, *_ = self.tts_module(**batch)

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(feats, feats_lengths)

        return

    def asr_encode():
        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        if self.encoder.interctc_use_conditioning:
            encoder_out, encoder_out_lens, _ = self.encoder(
                feats, feats_lengths, ctc=self.ctc
            )
        else:
            encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]


        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        if (
            getattr(self.encoder, "selfattention_layer_type", None) != "lf_selfattn"
            and not self.is_encoder_whisper
        ):
            assert encoder_out.size(-2) <= encoder_out_lens.max(), (
                encoder_out.size(),
                encoder_out_lens.max(),
            )

        if intermediate_outs is not None:
            return (encoder_out, intermediate_outs), encoder_out_lens

        return encoder_out, encoder_out_lens
    
    def collect_feats(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Caclualte features and return them as a dict.

        Args:
            text (Tensor): Text index tensor (B, T_text).
            text_lengths (Tensor): Text length tensor (B,).

        Returns:
            Dict[str, Tensor]: Dict of features.

        """
        return {}

    def inference(
        self,
        text: torch.Tensor,
        speech: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        **decode_config,
    ) -> Dict[str, torch.Tensor]:
        """Caclualte features and return them as a dict.

        Args:
            text (Tensor): Text index tensor (T_text).
            speech (Tensor): Speech waveform tensor (T_wav).
            spembs (Optional[Tensor]): Speaker embedding tensor (D,).
            sids (Optional[Tensor]): Speaker ID tensor (1,).
            lids (Optional[Tensor]): Language ID tensor (1,).
            durations (Optional[Tensor): Duration tensor.
            pitch (Optional[Tensor): Pitch tensor.
            energy (Optional[Tensor): Energy tensor.

        Returns:
            Dict[str, Tensor]: Dict of outputs.

        """
        input_dict = dict(text=text)
        if decode_config["use_teacher_forcing"] or getattr(self.tts, "use_gst", False):
            if speech is None:
                raise RuntimeError("missing required argument: 'speech'")
            if self.feats_extract is not None:
                feats = self.feats_extract(speech[None])[0][0]
            else:
                # Use precalculated feats (feats_type != raw case)
                feats = speech
            if self.normalize is not None:
                feats = self.normalize(feats[None])[0][0]
            input_dict.update(feats=feats)
            if self.tts.require_raw_speech:
                input_dict.update(speech=speech)

        if decode_config["use_teacher_forcing"]:
            if durations is not None:
                input_dict.update(durations=durations)

            if self.pitch_extract is not None:
                pitch = self.pitch_extract(
                    speech[None],
                    feats_lengths=torch.LongTensor([len(feats)]),
                    durations=durations[None],
                )[0][0]
            if self.pitch_normalize is not None:
                pitch = self.pitch_normalize(pitch[None])[0][0]
            if pitch is not None:
                input_dict.update(pitch=pitch)

            if self.energy_extract is not None:
                energy = self.energy_extract(
                    speech[None],
                    feats_lengths=torch.LongTensor([len(feats)]),
                    durations=durations[None],
                )[0][0]
            if self.energy_normalize is not None:
                energy = self.energy_normalize(energy[None])[0][0]
            if energy is not None:
                input_dict.update(energy=energy)

        if spembs is not None:
            input_dict.update(spembs=spembs)
        if sids is not None:
            input_dict.update(sids=sids)
        if lids is not None:
            input_dict.update(lids=lids)

        output_dict = self.tts.inference(**input_dict, **decode_config)

        if self.normalize is not None and output_dict.get("feat_gen") is not None:
            # NOTE: normalize.inverse is in-place operation
            feat_gen_denorm = self.normalize.inverse(
                output_dict["feat_gen"].clone()[None]
            )[0][0]
            output_dict.update(feat_gen_denorm=feat_gen_denorm)

        return output_dict
