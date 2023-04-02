"""Unsupervised (informed by an ASR model) Text-to-speech task."""

import argparse
import logging
from pathlib import Path
from typing import Callable, Collection, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from typeguard import check_argument_types, check_return_type

from espnet2.gan_tts.jets import JETS
from espnet2.gan_tts.joint import JointText2Wav
from espnet2.gan_tts.vits import VITS
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.espnet_model import ESPnetTTSModel
from espnet2.tts.fastspeech import FastSpeech
from espnet2.tts.fastspeech2 import FastSpeech2
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet2.tts.feats_extract.dio import Dio
from espnet2.tts.feats_extract.energy import Energy
from espnet2.tts.feats_extract.linear_spectrogram import LinearSpectrogram
from espnet2.tts.feats_extract.log_mel_fbank import LogMelFbank
from espnet2.tts.feats_extract.log_spectrogram import LogSpectrogram
from espnet2.tts.prodiff import ProDiff
from espnet2.tts.tacotron2 import Tacotron2
from espnet2.tts.transformer import Transformer
from espnet2.tts.utils import ParallelWaveGANPretrainedVocoder
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.griffin_lim import Spectrogram2Waveform
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.read_yaml_action import ReadYAMLAction
from espnet2.utils.types import int_or_none, str2bool, str_or_none

# FOR ASR
from espnet2.asr.ctc import CTC
from espnet2.tasks.asr import ASRTask
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.hugging_face_transformers_decoder import (  # noqa: H301
    HuggingFaceTransformersDecoder,
)
from espnet2.asr.decoder.mlm_decoder import MLMDecoder
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.decoder.s4_decoder import S4Decoder
from espnet2.asr.decoder.transducer_decoder import TransducerDecoder
from espnet2.asr.decoder.transformer_decoder import (
    DynamicConvolution2DTransformerDecoder,
    DynamicConvolutionTransformerDecoder,
    LightweightConvolution2DTransformerDecoder,
    LightweightConvolutionTransformerDecoder,
    TransformerDecoder,
)
from espnet2.asr.decoder.whisper_decoder import OpenAIWhisperDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.branchformer_encoder import BranchformerEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.contextual_block_conformer_encoder import (
    ContextualBlockConformerEncoder,
)
from espnet2.asr.encoder.contextual_block_transformer_encoder import (
    ContextualBlockTransformerEncoder,
)
from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder
from espnet2.asr.encoder.hubert_encoder import (
    FairseqHubertEncoder,
    FairseqHubertPretrainEncoder,
    TorchAudioHuBERTPretrainEncoder,
)
from espnet2.asr.encoder.longformer_encoder import LongformerEncoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.transformer_encoder_multispkr import (
    TransformerEncoder as TransformerEncoderMultiSpkr,
)
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.encoder.wav2vec2_encoder import FairSeqWav2Vec2Encoder
from espnet2.asr.encoder.whisper_encoder import OpenAIWhisperEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.fused import FusedFrontends
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.whisper import WhisperFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.maskctc_model import MaskCTCModel
from espnet2.asr.pit_espnet_model import ESPnetASRModel as PITESPnetModel
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.postencoder.hugging_face_transformers_postencoder import (
    HuggingFaceTransformersPostEncoder,
)
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.asr.preencoder.sinc import LightweightSincConvs
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.asr_transducer.joint_network import JointNetwork

# UTTS
from espnet2.utts.espnet_model import ESPnetUTTSModel
from espnet2.utts.tts.abs_utts_tts import AbsUTTSTTS
from espnet2.utts.tts.transformer import Transformer

# Token -> Mels
tts_choices = ClassChoices(
    "tts",
    classes=dict(
        transformer=Transformer,
    ),
    type_check=AbsUTTSTTS,
    default="transformer",
)

# Mels -> Token is handled by the ASR model, which is just defined by a tag.

class UTTSTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1


    # Add variable objects configurations
    class_choices_list = [
        # --tts and --tts_conf
        tts_choices
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

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
            "--tts_model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetTTSModel),
            help="The keyword arguments for model class.",
        )
        # group.add_argument(
        #     "--tts_model_file",
        #     type=str_or_none,
        #     default=None
        #     help="The file for an initial tts model.",
        # )

        # with arg.asr_model_config.open("r", encoding="utf-8") as f:
        #     asr_args = yaml.safe_load(f)
        # args = Namespace(**vars(args), **vars(asr_args))

        # ASR Options
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

        # if the asr_model_tag is None, the rest of the model can be defined
        # explicitly. Perhaps that would be useful if training your own asr
        # model. I'm leaving that unimplemented because I don't want to train my
        # own. So None actually raises an error.

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        return CommonCollateFn(
            float_pad_value=0.0,
            int_pad_value=0,
            not_sequence=["spembs", "sids", "lids"],
        )

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            retval = CommonPreprocessor(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
            )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        # if not inference:
        #     retval = ("text")
        # else:
        #     raise RuntimeError("No inference for utts.")
        #     retval = ("text",)
        return ("text",)

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = (
                "spembs",
                "durations",
                "pitch",
                "energy",
                "sids",
                "lids",
            )
        else:
            # Inference mode
            retval = (
                "spembs",
                "speech",
                "durations",
                "pitch",
                "energy",
                "sids",
                "lids",
            )
        return retval

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

        # THE ASR PART
        device = "cuda" if args.ngpu > 0 else "cpu"
        asr_model, asr_train_args = ASRTask.build_model_from_file(
            args.asr_model_config, args.asr_model_file, device
        )

        speech_feature_dim = asr_model.frontend.logmel.mel_options["n_mels"]

        # TODO: check some more things...
        assert args.token_list == asr_train_args.token_list
        # assert args.token_list == asr_train_args.token_list

        # THE TTS PART
        tts_class = tts_choices.get_class(args.tts)
        tts = tts_class(idim=vocab_size, odim=speech_feature_dim, **args.tts_conf)

        # LOSS
        ctc = CTC(
            odim=vocab_size, encoder_output_size=asr_train_args.encoder_conf["output_size"], **args.ctc_conf
        )

        # 5. Build model
        model = ESPnetUTTSModel(
            tts_module=tts,
            asr_normalize=asr_model.normalize,
            asr_preencoder=asr_model.preencoder,
            asr_encoder=asr_model.encoder,
            asr_postencoder=asr_model.postencoder,
            asr_decoder=asr_model.decoder,
            ctc=ctc,
            # token_list=token_list,
            # **args.model_conf,
        )

        assert check_return_type(model)

        print(args)

        # FIXME(kamo): Should be done in model?
        # 8. Initialize
        # if args.init is not None:
        #     initialize(model, args.init)

        return model
