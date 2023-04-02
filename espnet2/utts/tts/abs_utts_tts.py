# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Text-to-speech abstrast class."""

from abc import ABC, abstractmethod
from espnet2.tts.abs_tts import AbsTTS

class AbsUTTSTTS(AbsTTS):
    """An abstract class for TTS modules for pretraining with UTTS."""
    pass
