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

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.rnn.attentions import AttForwardDurationControl
from espnet.nets.pytorch_backend.tacotron2.encoder import encoder_init, Encoder
from espnet.nets.pytorch_backend.tacotron2.decoder import ZoneOutCell, Prenet, Postnet, decoder_init
from espnet.nets.pytorch_backend.e2e_tts_tacotron2 import (
    GuidedAttentionLoss,
    Tacotron2Loss,
)

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

class Decoder(torch.nn.Module):
    """Decoder module of Spectrogram prediction network.

    This is a module of decoder of Spectrogram prediction network in Tacotron2,
    which described in `Natural TTS
    Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`_.
    The decoder generates the sequence of
    features from the sequence of the hidden states.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884

    """

    def __init__(
        self,
        idim,
        odim,
        att,
        dlayers=2,
        dunits=1024,
        prenet_layers=2,
        prenet_units=256,
        postnet_layers=5,
        postnet_chans=512,
        postnet_filts=5,
        output_activation_fn=None,
        cumulate_att_w=True,
        use_batch_norm=True,
        use_concate=True,
        dropout_rate=0.5,
        zoneout_rate=0.1,
        reduction_factor=1,
    ):
        """Initialize Tacotron2 decoder module.

        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            att (torch.nn.Module): Instance of attention class.
            dlayers (int, optional): The number of decoder lstm layers.
            dunits (int, optional): The number of decoder lstm units.
            prenet_layers (int, optional): The number of prenet layers.
            prenet_units (int, optional): The number of prenet units.
            postnet_layers (int, optional): The number of postnet layers.
            postnet_filts (int, optional): The number of postnet filter size.
            postnet_chans (int, optional): The number of postnet filter channels.
            output_activation_fn (torch.nn.Module, optional):
                Activation function for outputs.
            cumulate_att_w (bool, optional):
                Whether to cumulate previous attention weight.
            use_batch_norm (bool, optional): Whether to use batch normalization.
            use_concate (bool, optional): Whether to concatenate encoder embedding
                with decoder lstm outputs.
            dropout_rate (float, optional): Dropout rate.
            zoneout_rate (float, optional): Zoneout rate.
            reduction_factor (int, optional): Reduction factor.

        """
        super(Decoder, self).__init__()

        # store the hyperparameters
        self.idim = idim
        self.odim = odim
        self.att = att
        self.output_activation_fn = output_activation_fn
        self.cumulate_att_w = cumulate_att_w
        self.use_concate = use_concate
        self.reduction_factor = reduction_factor

        # define lstm network
        prenet_units = prenet_units if prenet_layers != 0 else odim
        self.lstm = torch.nn.ModuleList()
        for layer in range(dlayers):
            iunits = idim + prenet_units if layer == 0 else dunits
            lstm = torch.nn.LSTMCell(iunits, dunits)
            if zoneout_rate > 0.0:
                lstm = ZoneOutCell(lstm, zoneout_rate)
            self.lstm += [lstm]

        # define prenet
        if prenet_layers > 0:
            self.prenet = Prenet(
                idim=odim,
                n_layers=prenet_layers,
                n_units=prenet_units,
                dropout_rate=dropout_rate,
            )
        else:
            self.prenet = None

        # define postnet
        if postnet_layers > 0:
            self.postnet = Postnet(
                idim=idim,
                odim=odim,
                n_layers=postnet_layers,
                n_chans=postnet_chans,
                n_filts=postnet_filts,
                use_batch_norm=use_batch_norm,
                dropout_rate=dropout_rate,
            )
        else:
            self.postnet = None

        # define projection layers
        iunits = idim + dunits if use_concate else dunits
        self.feat_out = torch.nn.Linear(iunits, odim * reduction_factor, bias=False)
        self.prob_out = torch.nn.Linear(iunits, reduction_factor)

        # initialize
        self.apply(decoder_init)

    def _zero_state(self, hs):
        init_hs = hs.new_zeros(hs.size(0), self.lstm[0].hidden_size)
        return init_hs

    def forward(self, q, hs, hlens, ys):
        """Calculate forward propagation.

        Args:
            q (Tensor): Batch of the outputs from the duration encoder (B, Tmax)
            hs (Tensor): Batch of the sequences of padded hidden states (B, Tmax, idim).
            hlens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor):
                Batch of the sequences of padded target features (B, Lmax, odim).

        Returns:
            Tensor: Batch of output tensors after postnet (B, Lmax, odim).
            Tensor: Batch of output tensors before postnet (B, Lmax, odim).
            Tensor: Batch of logits of stop prediction (B, Lmax).
            Tensor: Batch of attention weights (B, Lmax, Tmax).

        Note:
            This computation is performed in teacher-forcing manner.

        """
        # thin out frames (B, Lmax, odim) ->  (B, Lmax/r, odim)
        if self.reduction_factor > 1:
            ys = ys[:, self.reduction_factor - 1 :: self.reduction_factor]

        # length list should be list of int
        hlens = list(map(int, hlens))

        # initialize hidden states of decoder
        c_list = [self._zero_state(hs)]
        z_list = [self._zero_state(hs)]
        for _ in range(1, len(self.lstm)):
            c_list += [self._zero_state(hs)]
            z_list += [self._zero_state(hs)]
        prev_out = hs.new_zeros(hs.size(0), self.odim)

        # initialize attention
        prev_att_w = None
        self.att.reset()

        # loop for an output sequence
        outs, logits, att_ws = [], [], []
        for y in ys.transpose(0, 1):
            att_c, att_w = self.att(q, hs, hlens, z_list[0], prev_att_w)
            prenet_out = self.prenet(prev_out) if self.prenet is not None else prev_out
            xs = torch.cat([att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for i in range(1, len(self.lstm)):
                z_list[i], c_list[i] = self.lstm[i](
                    z_list[i - 1], (z_list[i], c_list[i])
                )
            zcs = (
                torch.cat([z_list[-1], att_c], dim=1)
                if self.use_concate
                else z_list[-1]
            )
            outs += [self.feat_out(zcs).view(hs.size(0), self.odim, -1)]
            logits += [self.prob_out(zcs)]
            att_ws += [att_w]
            prev_out = y  # teacher forcing
            if self.cumulate_att_w and prev_att_w is not None:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
            else:
                prev_att_w = att_w

        logits = torch.cat(logits, dim=1)  # (B, Lmax)
        before_outs = torch.cat(outs, dim=2)  # (B, odim, Lmax)
        att_ws = torch.stack(att_ws, dim=1)  # (B, Lmax, Tmax)

        if self.reduction_factor > 1:
            before_outs = before_outs.view(
                before_outs.size(0), self.odim, -1
            )  # (B, odim, Lmax)

        if self.postnet is not None:
            after_outs = before_outs + self.postnet(before_outs)  # (B, odim, Lmax)
        else:
            after_outs = before_outs
        before_outs = before_outs.transpose(2, 1)  # (B, Lmax, odim)
        after_outs = after_outs.transpose(2, 1)  # (B, Lmax, odim)
        logits = logits

        # apply activation function for scaling
        if self.output_activation_fn is not None:
            before_outs = self.output_activation_fn(before_outs)
            after_outs = self.output_activation_fn(after_outs)

        return after_outs, before_outs, logits, att_ws

    def inference(
            self,
            q,
            h,
            threshold=0.5,
            minlenratio=0.0,
            maxlenratio=10.0,
            use_att_constraint=False,
            backward_window=None,
            forward_window=None,
    ):
        """Generate the sequence of features given the sequences of characters.

        Args:
            h (Tensor): Input sequence of encoder hidden states (T, C).
            threshold (float, optional): Threshold to stop generation.
            minlenratio (float, optional): Minimum length ratio.
                If set to 1.0 and the length of input is 10,
                the minimum length of outputs will be 10 * 1 = 10.
            minlenratio (float, optional): Minimum length ratio.
                If set to 10 and the length of input is 10,
                the maximum length of outputs will be 10 * 10 = 100.
            use_att_constraint (bool):
                Whether to apply attention constraint introduced in `Deep Voice 3`_.
            backward_window (int): Backward window size in attention constraint.
            forward_window (int): Forward window size in attention constraint.

        Returns:
            Tensor: Output sequence of features (L, odim).
            Tensor: Output sequence of stop probabilities (L,).
            Tensor: Attention weights (L, T).

        Note:
            This computation is performed in auto-regressive manner.

        .. _`Deep Voice 3`: https://arxiv.org/abs/1710.07654

        """
        # setup
        assert len(h.size()) == 2
        hs = h.unsqueeze(0)
        ilens = [h.size(0)]
        maxlen = int(h.size(0) * maxlenratio)
        minlen = int(h.size(0) * minlenratio)

        # initialize hidden states of decoder
        c_list = [self._zero_state(hs)]
        z_list = [self._zero_state(hs)]
        for _ in range(1, len(self.lstm)):
            c_list += [self._zero_state(hs)]
            z_list += [self._zero_state(hs)]
        prev_out = hs.new_zeros(1, self.odim)

        # initialize attention
        prev_att_w = None
        self.att.reset()

        # setup for attention constraint
        if use_att_constraint:
            last_attended_idx = 0
        else:
            last_attended_idx = None

        # loop for an output sequence
        idx = 0
        outs, att_ws, probs = [], [], []
        while True:
            # updated index
            idx += self.reduction_factor

            # decoder calculation
            att_c, att_w = self.att(
                q,
                hs,
                ilens,
                z_list[0],
                prev_att_w,
                last_attended_idx=last_attended_idx,
                backward_window=backward_window,
                forward_window=forward_window,
            )

            att_ws += [att_w]
            prenet_out = self.prenet(prev_out) if self.prenet is not None else prev_out
            xs = torch.cat([att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for i in range(1, len(self.lstm)):
                z_list[i], c_list[i] = self.lstm[i](
                    z_list[i - 1], (z_list[i], c_list[i])
                )
            zcs = (
                torch.cat([z_list[-1], att_c], dim=1)
                if self.use_concate
                else z_list[-1]
            )
            outs += [self.feat_out(zcs).view(1, self.odim, -1)]  # [(1, odim, r), ...]
            probs += [torch.sigmoid(self.prob_out(zcs))[0]]  # [(r), ...]
            if self.output_activation_fn is not None:
                prev_out = self.output_activation_fn(outs[-1][:, :, -1])  # (1, odim)
            else:
                prev_out = outs[-1][:, :, -1]  # (1, odim)
            if self.cumulate_att_w and prev_att_w is not None:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
            else:
                prev_att_w = att_w
            if use_att_constraint:
                last_attended_idx = int(att_w.argmax())

            # check whether to finish generation
            if int(sum(probs[-1] >= threshold)) > 0 or idx >= maxlen:
                # check mininum length
                if idx < minlen:
                    continue
                outs = torch.cat(outs, dim=2)  # (1, odim, L)
                if self.postnet is not None:
                    outs = outs + self.postnet(outs)  # (1, odim, L)
                outs = outs.transpose(2, 1).squeeze(0)  # (L, odim)
                probs = torch.cat(probs, dim=0)
                att_ws = torch.cat(att_ws, dim=0)
                break

        if self.output_activation_fn is not None:
            outs = self.output_activation_fn(outs)

        return outs, probs, att_ws

    def calculate_all_attentions(self, hs, hlens, ys):
        """Calculate all of the attention weights.

        Args:
            hs (Tensor): Batch of the sequences of padded hidden states (B, Tmax, idim).
            hlens (LongTensor): Batch of lengths of each input batch (B,).
            ys (Tensor):
                Batch of the sequences of padded target features (B, Lmax, odim).

        Returns:
            numpy.ndarray: Batch of attention weights (B, Lmax, Tmax).

        Note:
            This computation is performed in teacher-forcing manner.

        """
        # thin out frames (B, Lmax, odim) ->  (B, Lmax/r, odim)
        if self.reduction_factor > 1:
            ys = ys[:, self.reduction_factor - 1 :: self.reduction_factor]

        # length list should be list of int
        hlens = list(map(int, hlens))

        # initialize hidden states of decoder
        c_list = [self._zero_state(hs)]
        z_list = [self._zero_state(hs)]
        for _ in range(1, len(self.lstm)):
            c_list += [self._zero_state(hs)]
            z_list += [self._zero_state(hs)]
        prev_out = hs.new_zeros(hs.size(0), self.odim)

        # initialize attention
        prev_att_w = None
        self.att.reset()

        # loop for an output sequence
        att_ws = []
        for y in ys.transpose(0, 1):
            if self.use_att_extra_inputs:
                att_c, att_w = self.att(hs, hlens, z_list[0], prev_att_w, prev_out)
            else:
                att_c, att_w = self.att(hs, hlens, z_list[0], prev_att_w)
            att_ws += [att_w]
            prenet_out = self.prenet(prev_out) if self.prenet is not None else prev_out
            xs = torch.cat([att_c, prenet_out], dim=1)
            z_list[0], c_list[0] = self.lstm[0](xs, (z_list[0], c_list[0]))
            for i in range(1, len(self.lstm)):
                z_list[i], c_list[i] = self.lstm[i](
                    z_list[i - 1], (z_list[i], c_list[i])
                )
            prev_out = y  # teacher forcing
            if self.cumulate_att_w and prev_att_w is not None:
                prev_att_w = prev_att_w + att_w  # Note: error when use +=
            else:
                prev_att_w = att_w

        att_ws = torch.stack(att_ws, dim=1)  # (B, Lmax, Tmax)

        return att_ws

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
            eunits: int = 512,
            econv_layers: int = 3,
            econv_chans: int = 512,
            econv_filts: int = 5,
            use_residual: bool = False,
            ## duration encoder:
            ## decoder:
            aconv_chans: int = 32,
            aconv_filts: int = 15,
            dlayers: int = 6,
            dunits: int = 1536,
            prenet_layers: int = 2,
            prenet_units: int = 256,
            postnet_layers: int = 5,
            postnet_chans: int = 512,
            postnet_filts: int = 5,
            output_activation: str = None,
            cumulate_att_w: bool = True,
            use_concat: bool = True,

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
            dropout_rate: float = 0.5,
            zoneout_rate: float = 0.1,
            use_masking: bool = True,
            use_weighted_masking: bool = False,
            bce_pos_weight: float = 5.0,
            loss_type: str = "L1+L2",
            use_guided_attn_loss: bool = True,
            guided_attn_loss_sigma: float = 0.4,
            guided_attn_loss_lambda: float = 1.0,
            init_type: str = "xavier_uniform",
            init_enc_alpha: float = 1.0,
            init_dec_alpha: float = 1.0,
            loss_function: str = "XiaoiceSing2",  # FastSpeech1, XiaoiceSing2
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
        self.cumulate_att_w = cumulate_att_w
        self.reduction_factor = reduction_factor
        self.use_guided_attn_loss = use_guided_attn_loss
        self.loss_type = loss_type

        self.midi_embed_integration_type = midi_embed_integration_type

        # define activation function for the final output
        if output_activation is None:
            self.output_activation_fn = None
        elif hasattr(F, output_activation):
            self.output_activation_fn = getattr(F, output_activation)
        else:
            raise ValueError(
                f"there is no such an activation function. " f"({output_activation})"
            )

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

        dec_idim = eunits
        att = AttForwardDurationControl(dec_idim, dunits, adim, aconv_chans, aconv_filts)

        self.dec = Decoder(
            idim=dec_idim,
            odim=odim,
            att=att,
            dlayers=dlayers,
            dunits=dunits,
            prenet_layers=prenet_layers,
            prenet_units=prenet_units,
            postnet_layers=postnet_layers,
            postnet_chans=postnet_chans,
            postnet_filts=postnet_filts,
            output_activation_fn=self.output_activation_fn,
            cumulate_att_w=self.cumulate_att_w,
            use_batch_norm=use_batch_norm,
            use_concate=use_concat,
            dropout_rate=dropout_rate,
            zoneout_rate=zoneout_rate,
            reduction_factor=reduction_factor,
        )

        self.taco2_loss = Tacotron2Loss(
            use_masking=use_masking,
            use_weighted_masking=use_weighted_masking,
            bce_pos_weight=bce_pos_weight,
        )

        if self.use_guided_attn_loss:
            self.attn_loss = GuidedAttentionLoss(
                sigma=guided_attn_loss_sigma,
                alpha=guided_attn_loss_lambda,
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

        # useful variables
        ys = feats
        olens = feats_lengths

        # make labels for stop prediction
        labels = make_pad_mask(olens - 1).to(ys.device, ys.dtype)
        labels = F.pad(labels, [0, 1], "constant", 1.0)

        # decoder
        after_outs, before_outs, logits, att_ws = self.dec(q, hs, hlens, ys)

        # modify mod part of groundtruth
        if self.reduction_factor > 1:
            assert olens.ge(
                self.reduction_factor
            ).all(), "Output length must be greater than or equal to reduction factor."
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            max_out = max(olens)
            ys = ys[:, :max_out]
            labels = labels[:, :max_out]
            labels = torch.scatter(
                labels, 1, (olens - 1).unsqueeze(1), 1.0
            )  # see #3388

        # calculate taco2 loss
        l1_loss, mse_loss, bce_loss = self.taco2_loss(
            after_outs, before_outs, logits, ys, labels, olens
        )

        if self.loss_type == "L1+L2":
            loss = l1_loss + mse_loss + bce_loss
        elif self.loss_type == "L1":
            loss = l1_loss + bce_loss
        elif self.loss_type == "L2":
            loss = mse_loss + bce_loss
        else:
            raise ValueError(f"unknown --loss-type {self.loss_type}")

        stats = dict(
            l1_loss=l1_loss.item(),
            mse_loss=mse_loss.item(),
            bce_loss=bce_loss.item(),
        )

        # calculate attention loss
        if self.use_guided_attn_loss:
            # NOTE(kan-bayashi): length of output for auto-regressive
            # input will be changed when r > 1
            if self.reduction_factor > 1:
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                olens_in = olens
            attn_loss = self.attn_loss(att_ws, ilens, olens_in)
            loss = loss + attn_loss
            stats.update(attn_loss=attn_loss.item())

        if not joint_training:
            stats.update(loss=loss.item())
            loss, stats, weight = force_gatherable(
                (loss, stats, batch_size), loss.device
            )
            return loss, stats, weight
        else:
            return loss, stats, after_outs

    def inference(
            self,
            text: torch.Tensor,
            **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Return predicted output as a dict."""
        raise NotImplementedError
