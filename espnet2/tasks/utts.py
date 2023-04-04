"""Text-to-speech task with ASR component."""

import argparse
import logging
from pathlib import Path
from typing import Callable, Collection, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from typeguard import check_argument_types, check_return_type

from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.read_yaml_action import ReadYAMLAction
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none

from espnet2.tts.feats_extract.log_mel_fbank import LogMelFbank
from espnet2.tts.espnet_model import ESPnetTTSModel
from espnet2.tasks.tts import (
    TTSTask, feats_extractor_choices, pitch_extractor_choices, energy_extractor_choices, normalize_choices, pitch_normalize_choices, energy_normalize_choices, tts_choices
)
from espnet2.tasks.asr import ASRTask

from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.utts.espnet_model import ESPnetUTTSModel

class UTTSTask(TTSTask):
    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
        assert check_argument_types()
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["token_list"]

        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetTTSModel),
            help="The keyword arguments for model class.",
        )

        # group = parser.add_argument_group(description="ASR related")
        group.add_argument(
            "--asr_model_config",
            action=ReadYAMLAction,
            help="The config file for the asr model, from which the rest of the model can be inferred.",
        )

        group.add_argument(
            "--asr_model_file",
            type=str_or_none,
            default=None,
            help="The file for the asr model parameters.",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetUTTSModel:
        assert check_argument_types()
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line[0] + line[1:].rstrip() for line in f]

            # "args" is saved as it is in a yaml file by BaseTask.main().
            # Overwriting token_list to keep it as "portable".
            args.token_list = token_list.copy()
        elif isinstance(args.token_list, (tuple, list)):
            token_list = args.token_list.copy()
        else:
            raise RuntimeError("token_list must be str or dict")

        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")

        # ASR SECTION
        # 1.
        device = "cuda" if args.ngpu > 0 else "cpu"
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            args.asr_model_config, args.asr_model_file, device
        )

        assert token_list == asr_train_args.token_list

        # TTS SECTION
        # 1. feats_extract
        # this must match the front end of the asr model. right now craft it manually.
        n_fft = asr_model.frontend.stft.n_fft
        win_length = asr_model.frontend.stft.win_length
        hop_length = asr_model.frontend.stft.hop_length
        window = asr_model.frontend.stft.window
        center = asr_model.frontend.stft.center
        normalized = asr_model.frontend.stft.normalized
        onesided = asr_model.frontend.stft.onesided

        fs = asr_model.frontend.logmel.mel_options["sr"]
        n_mels = asr_model.frontend.logmel.mel_options["n_mels"]
        fmin = int(asr_model.frontend.logmel.mel_options["fmin"])
        fmax = int(asr_model.frontend.logmel.mel_options["fmax"])
        htk = asr_model.frontend.logmel.mel_options["htk"]
        log_base = asr_model.frontend.logmel.log_base

        feats_extract = LogMelFbank(
            fs = fs,
            n_fft = n_fft,
            win_length = win_length,
            hop_length = hop_length,
            window = window,
            center = center,
            normalized = normalized,
            onesided = onesided,
            n_mels = n_mels,
            fmin = fmin,
            fmax = fmax,
            htk = htk,
            log_base = log_base,
        )
        odim = feats_extract.output_size()

        # 2. Normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # 3. TTS
        tts_class = tts_choices.get_class(args.tts)
        tts = tts_class(idim=vocab_size, odim=odim, **args.tts_conf)

        # 4. Extra components
        pitch_extract = None
        energy_extract = None
        pitch_normalize = None
        energy_normalize = None
        if getattr(args, "pitch_extract", None) is not None:
            pitch_extract_class = pitch_extractor_choices.get_class(args.pitch_extract)
            if args.pitch_extract_conf.get("reduction_factor", None) is not None:
                assert args.pitch_extract_conf.get(
                    "reduction_factor", None
                ) == args.tts_conf.get("reduction_factor", 1)
            else:
                args.pitch_extract_conf["reduction_factor"] = args.tts_conf.get(
                    "reduction_factor", 1
                )
                pitch_extract = pitch_extract_class(**args.pitch_extract_conf)
        if getattr(args, "energy_extract", None) is not None:
            if args.energy_extract_conf.get("reduction_factor", None) is not None:
                assert args.energy_extract_conf.get(
                    "reduction_factor", None
                ) == args.tts_conf.get("reduction_factor", 1)
            else:
                args.energy_extract_conf["reduction_factor"] = args.tts_conf.get(
                    "reduction_factor", 1
                )
                energy_extract_class = energy_extractor_choices.get_class(
                    args.energy_extract
                )
                energy_extract = energy_extract_class(**args.energy_extract_conf)
        if getattr(args, "pitch_normalize", None) is not None:
            pitch_normalize_class = pitch_normalize_choices.get_class(
                args.pitch_normalize
            )
            pitch_normalize = pitch_normalize_class(**args.pitch_normalize_conf)
        if getattr(args, "energy_normalize", None) is not None:
            energy_normalize_class = energy_normalize_choices.get_class(
                args.energy_normalize
            )
            energy_normalize = energy_normalize_class(**args.energy_normalize_conf)

        # 5. Build model
        model = ESPnetUTTSModel(
            feats_extract=feats_extract,
            pitch_extract=pitch_extract,
            energy_extract=energy_extract,
            normalize=normalize,
            pitch_normalize=pitch_normalize,
            energy_normalize=energy_normalize,
            tts=tts,
            # asr model componentsh
            asr_model=asr_model,
            **args.model_conf,
        )
        assert check_return_type(model)
        return model

    # ~~~~~~~~~ The methods below are mainly used for inference ~~~~~~~~~
    @classmethod
    def build_model_from_file(
        cls,
        config_file: Union[Path, str] = None,
        model_file: Union[Path, str] = None,
        device: str = "cpu",
    ) -> Tuple[AbsESPnetModel, argparse.Namespace]:
        """Build model from the files.

        This method is used for inference or fine-tuning.

        Args:
            config_file: The yaml file saved when training.
            model_file: The model file saved when training.
            device: Device type, "cpu", "cuda", or "cuda:N".

        """
        assert check_argument_types()
        if config_file is None:
            assert model_file is not None, (
                "The argument 'model_file' must be provided "
                "if the argument 'config_file' is not specified."
            )
            config_file = Path(model_file).parent / "config.yaml"
        else:
            config_file = Path(config_file)

        with config_file.open("r", encoding="utf-8") as f:
            args = yaml.safe_load(f)
        args = argparse.Namespace(**args)
        utts_model = cls.build_model(args)
        if not isinstance(utts_model, AbsESPnetModel):
            raise RuntimeError(
                f"model must inherit {AbsESPnetModel.__name__}, but got {type(utts_model)}"
            )

        if model_file is not None:
            if device == "cuda":
                # NOTE(kamo): "cuda" for torch.load always indicates cuda:0
                #   in PyTorch<=1.4
                device = f"cuda:{torch.cuda.current_device()}"
            utts_model.load_state_dict(torch.load(model_file))

        model = TTSTask.build_model_from_utts_model(cls, utts_model, args)
        model.to(device)

        return model, args
