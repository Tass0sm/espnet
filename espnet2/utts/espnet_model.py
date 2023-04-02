"""Text-to-speech ESPnet model with asr component."""

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
from espnet2.tts.espnet_model import ESPnetTTSModel

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
            asr_normalize: Optional[AbsNormalize],
            asr_preencoder: Optional[AbsPreEncoder],
            asr_encoder: AbsEncoder,
            asr_postencoder: Optional[AbsPostEncoder],
            asr_decoder: Optional[AbsDecoder],
            asr_ctc: Optional[CTC],
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
        loss, stats, after_outs = super().forward(
            text,
            text_lengths,
            speech,
            speech_lengths,
            durations,
            durations_lengths,
            pitch,
            pitch_lengths,
            energy,
            energy_lengths,
            spembs,
            sids,
            lids,
            **kwargs
        )

        print("SPEECH", speech.shape)

        # print("LOSS", loss)
        # print("STATS", stats)
        # print("AFTER_OUTS", after_outs)

        breakpoint()

        return loss, stats, after_outs
