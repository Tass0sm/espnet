# Copyright 2023 Tassos Manganaris
#  TODO: License

"""Singing-Tacotron related modules for ESPnet2."""

import logging
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from typeguard import check_argument_types

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.svs.abs_svs import AbsSVS

from espnet.nets.pytorch_backend.rnn.attentions import AttForward, AttForwardTA, AttLoc
from espnet.nets.pytorch_backend.tacotron2.encoder import encoder_init, Encoder
# from espnet.nets.pytorch_backend.tacotron2.decoder import Decoder

class DurationEncoder(torch.nn.Module):
    """Duration encoder module of Singing-Tacotron.

    TODO: Description

    .. _`Singing-Tacotron: Global duration control attention and dynamic filter
       for End-to-end singing voice synthesis`: https://arxiv.org/abs/2202.07907

    """

    def __init__(
            self,
            idim,
            embed_dim=512,
            econv_layers=2,
            econv_chans=32,
            econv_filts=5,
            use_batch_norm=True,
            dropout_rate=0.5,
            padding_idx=0,
    ):
        """Initialize DurationEncoder module.

        Args:

        """
        super(DurationEncoder, self).__init__()

        self.embed = torch.nn.Embedding(idim, embed_dim, padding_idx=padding_idx)
        self.fc1 = torch.nn.Linear(embed_dim, econv_chans)

        if econv_layers > 0:
            self.convs = torch.nn.ModuleList()
            for layer in range(econv_layers):
                ichans = econv_chans
                if use_batch_norm:
                    self.convs += [
                        torch.nn.Sequential(
                            torch.nn.Conv1d(
                                ichans,
                                econv_chans,
                                econv_filts,
                                stride=1,
                                padding=(econv_filts - 1) // 2,
                                bias=False,
                            ),
                            torch.nn.BatchNorm1d(econv_chans),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_rate),
                        )
                    ]
                else:
                    self.convs += [
                        torch.nn.Sequential(
                            torch.nn.Conv1d(
                                ichans,
                                econv_chans,
                                econv_filts,
                                stride=1,
                                padding=(econv_filts - 1) // 2,
                                bias=False,
                            ),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_rate),
                        )
                    ]
        else:
            self.convs = None

        iunits = econv_chans if econv_layers != 0 else embed_dim
        self.fc2 = torch.nn.Linear(iunits, 1)
        self.tanh = torch.nn.Tanh()

        # initialize
        self.apply(encoder_init)

    def forward(self, ds):
        """Calculate forward propagation.

        Args:
            TODO

        Returns:
            TODO

        """


        ds = self.embed(ds)
        ds = self.fc1(ds).transpose(1, 2)

        if self.convs is not None:
            for i in range(len(self.convs)):
                ds = self.convs[i](ds)

        ds = ds.transpose(1, 2)
        ds = self.fc2(ds)
        q = (self.tanh(ds) + 1) / 2
        return q.squeeze()



class SingingTacotron(AbsSVS):
    """
    TODO: Description

    .. _`Singing-Tacotron: Global duration control attention and dynamic filter
       for End-to-end singing voice synthesis`: https://arxiv.org/abs/2202.07907

    """

    def __init__(
            self,
            # network structure related
            idim: int,
            odim: int,
            midi_dim: int = 129,
            tempo_dim: int = 500,
            embed_dim: int = 512,
            adim: int = 384,
            aheads: int = 4,
            ## input prep
            midi_embed_integration_type: str = "concat",
            ## content encoder:
            elayers: int = 6,
            eunits: int = 1536,
            econv_layers: int = 3,
            econv_chans: int = 512,
            econv_filts: int = 5,
            use_residual: bool = False,
            dropout_rate: float = 0.1,
            ## duration encoder:
            ## decoder:
            dlayers: int = 6,
            dunits: int = 1536,
            postnet_layers: int = 5,
            postnet_chans: int = 512,
            postnet_filts: int = 5,
            postnet_dropout_rate: float = 0.5,
            positionwise_layer_type: str = "conv1d",
            positionwise_conv_kernel_size: int = 1,
            use_scaled_pos_enc: bool = True,
            use_batch_norm: bool = True,
            encoder_normalize_before: bool = True,
            decoder_normalize_before: bool = True,
            encoder_concat_after: bool = False,
            decoder_concat_after: bool = False,
            duration_predictor_layers: int = 2,
            duration_predictor_chans: int = 384,
            duration_predictor_kernel_size: int = 3,
            duration_predictor_dropout_rate: float = 0.1,
            reduction_factor: int = 1,
            encoder_type: str = "transformer",
            decoder_type: str = "transformer",
            transformer_enc_dropout_rate: float = 0.1,
            transformer_enc_positional_dropout_rate: float = 0.1,
            transformer_enc_attn_dropout_rate: float = 0.1,
            transformer_dec_dropout_rate: float = 0.1,
            transformer_dec_positional_dropout_rate: float = 0.1,
            transformer_dec_attn_dropout_rate: float = 0.1,
            # only for conformer
            conformer_rel_pos_type: str = "legacy",
            conformer_pos_enc_layer_type: str = "rel_pos",
            conformer_self_attn_layer_type: str = "rel_selfattn",
            conformer_activation_type: str = "swish",
            use_macaron_style_in_conformer: bool = True,
            use_cnn_in_conformer: bool = True,
            zero_triu: bool = False,
            conformer_enc_kernel_size: int = 7,
            conformer_dec_kernel_size: int = 31,
            # extra embedding related
            spks: Optional[int] = None,
            langs: Optional[int] = None,
            spk_embed_dim: Optional[int] = None,
            spk_embed_integration_type: str = "add",
            # training related
            init_type: str = "xavier_uniform",
            init_enc_alpha: float = 1.0,
            init_dec_alpha: float = 1.0,
            use_masking: bool = False,
            use_weighted_masking: bool = False,
            loss_function: str = "XiaoiceSing2",  # FastSpeech1, XiaoiceSing2
            loss_type: str = "L1",
            lambda_mel: float = 1,
            lambda_dur: float = 0.1,
            lambda_pitch: float = 0.01,
            lambda_vuv: float = 0.01,
    ):
        """Initialize SingingTacotron module.

        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            elayers (int): Number of encoder layers.
            eunits (int): Number of encoder hidden units.
            dlayers (int): Number of decoder layers.
            dunits (int): Number of decoder hidden units.
            postnet_layers (int): Number of postnet layers.
            postnet_chans (int): Number of postnet channels.
            postnet_filts (int): Kernel size of postnet.
            postnet_dropout_rate (float): Dropout rate in postnet.
            use_scaled_pos_enc (bool): Whether to use trainable scaled pos encoding.
            use_batch_norm (bool): Whether to use batch normalization in encoder prenet.
            encoder_normalize_before (bool): Whether to apply layernorm layer before
                encoder block.
            decoder_normalize_before (bool): Whether to apply layernorm layer before
                decoder block.
            encoder_concat_after (bool): Whether to concatenate attention layer's input
                and output in encoder.
            decoder_concat_after (bool): Whether to concatenate attention layer's input
                and output in decoder.
            duration_predictor_layers (int): Number of duration predictor layers.
            duration_predictor_chans (int): Number of duration predictor channels.
            duration_predictor_kernel_size (int): Kernel size of duration predictor.
            duration_predictor_dropout_rate (float): Dropout rate in duration predictor.
            reduction_factor (int): Reduction factor.
            encoder_type (str): Encoder type ("transformer" or "conformer").
            decoder_type (str): Decoder type ("transformer" or "conformer").
            transformer_enc_dropout_rate (float): Dropout rate in encoder except
                attention and positional encoding.
            transformer_enc_positional_dropout_rate (float): Dropout rate after encoder
                positional encoding.
            transformer_enc_attn_dropout_rate (float): Dropout rate in encoder
                self-attention module.
            transformer_dec_dropout_rate (float): Dropout rate in decoder except
                attention & positional encoding.
            transformer_dec_positional_dropout_rate (float): Dropout rate after decoder
                positional encoding.
            transformer_dec_attn_dropout_rate (float): Dropout rate in decoder
                self-attention module.
            spks (Optional[int]): Number of speakers. If set to > 1, assume that the
                sids will be provided as the input and use sid embedding layer.
            langs (Optional[int]): Number of languages. If set to > 1, assume that the
                lids will be provided as the input and use sid embedding layer.
            spk_embed_dim (Optional[int]): Speaker embedding dimension. If set to > 0,
                assume that spembs will be provided as the input.
            spk_embed_integration_type: How to integrate speaker embedding.
            init_type (str): How to initialize transformer parameters.
            init_enc_alpha (float): Initial value of alpha in scaled pos encoding of the
                encoder.
            init_dec_alpha (float): Initial value of alpha in scaled pos encoding of the
                decoder.
            use_masking (bool): Whether to apply masking for padded part in loss
                calculation.
            use_weighted_masking (bool): Whether to apply weighted masking in loss
                calculation.
            loss_function (str): Loss functions ("FastSpeech1" or "XiaoiceSing2")
            loss_type (str): Loss type ("L1" (MAE) or "L2" (MSE))
            lambda_mel (float): Loss scaling coefficient for Mel loss.
            lambda_dur (float): Loss scaling coefficient for duration loss.
            lambda_pitch (float): Loss scaling coefficient for pitch loss.
            lambda_vuv (float): Loss scaling coefficient for VUV loss.

        """
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1

        self.midi_embed_integration_type = midi_embed_integration_type

        padding_idx = 0
        self.padding_idx = padding_idx

        # for the content encoder input
        # embeddings
        self.label_embedding = torch.nn.Embedding(
            num_embeddings=idim, embedding_dim=eunits, padding_idx=self.padding_idx
        )
        self.midi_embedding = torch.nn.Embedding(
            num_embeddings=midi_dim, embedding_dim=eunits, padding_idx=self.padding_idx,
        )

        h_dim = eunits

        if midi_embed_integration_type == "concat":
            h_dim = 2 * eunits

        # define network modules
        self.content_encoder = Encoder(
            idim=h_dim,
            input_layer="linear",
            elayers=elayers,
            eunits=eunits,
            econv_layers=econv_layers,
            econv_chans=econv_chans,
            econv_filts=econv_filts,
            use_batch_norm=use_batch_norm,
            use_residual=use_residual,
            dropout_rate=dropout_rate,
            padding_idx=padding_idx,
        )

        # for the duration encoder
        self.duration_encoder = DurationEncoder(
            idim=tempo_dim
        )

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

        # Add eos at the last of sequence
        label = F.pad(label, [0, 1], "constant", self.padding_idx)
        for i, l in enumerate(label_lengths):
            label[i, l] = self.eos

        # Add eos at the last of sequence
        midi = F.pad(midi, [0, 1], "constant", self.padding_idx)
        for i, l in enumerate(midi_lengths):
            midi[i, l] = self.eos

        ilens = np.minimum(label_lengths, midi_lengths) + 1

        label_emb = self.label_embedding(label)  # FIX ME: label Float to Int
        midi_emb = self.midi_embedding(midi)

        if self.midi_embed_integration_type == "add":
            hs = label_emb + midi_emb
        else:
            hs = torch.cat((label_emb, midi_emb), dim=-1)

        hs, hlens = self.content_encoder(hs, ilens)

        # Add eos at the last of sequence
        tempo = F.pad(tempo, [0, 1], "constant", self.padding_idx)
        for i, l in enumerate(tempo_lengths):
            tempo[i, l] = self.eos

        ilens = np.minimum(label_lengths, midi_lengths) + 1

        q = self.duration_encoder(tempo)

        # decoder

        breakpoint()

        # hs = self.encoder(hs)

        # x = torch.cat((text, pitch), dim=-1)
        # print("X", x.shape)

        raise NotImplementedError

    def inference(
            self,
            text: torch.Tensor,
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Return predicted output as a dict."""
        raise NotImplementedError
