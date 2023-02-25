#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-Head Attention layer definition."""

import math

import torch
from torch import nn
import torch.nn.functional as F
import axial_attention.axial_attention as lucidrains

###############################################################################
#                              axial attention 2                              #
###############################################################################

class MultiHeadedLucidrainsAxialSelfAttentionWrapper(nn.Module):
    """Multi-Head Axial Attention layer.

    Args:
        dim
        shape
        num_dimensions = 2
        heads = 8
        dim_heads = None
        dim_index = -1
        sum_axial_out = True
    """

    def __init__(self, dim, shape, num_dimensions = 2, heads = 8, dim_heads = None, dim_index = -1, sum_axial_out = True):
        """Construct a MultiHeadedLucidrainsAxialSelfAttention object."""
        super(MultiHeadedLucidrainsAxialSelfAttentionWrapper, self).__init__()
        self.shape = shape
        self.forward_attention = lucidrains.AxialAttention(dim, num_dimensions, heads, dim_heads, dim_index, sum_axial_out)

    def forward(self, query_ignored, key, value_ignored, mask_ignored):
        """Wrapper for computing axial attention.

        Args:
            query (torch.Tensor): Query tensor ignored
            key (torch.Tensor): Key tensor (#batch, time2, original_size).
            value (torch.Tensor): Value tensor ignored.
            mask (torch.Tensor): Mask tensor ignored

        Returns:
            torch.Tensor: Output tensor (#batch, time1, original_size).

        """
        key_tensor = torch.unflatten(key, 2, self.shape)  # unflatten just for applying multihead
        out = self.forward_attention(key_tensor)
        out = torch.flatten(out, start_dim=2)
        return out

class MultiHeadedMedicalAxialSelfAttentionWrapper(nn.Module):
    """Multi-Head Axial Attention layer.

    Args:
        dim
        shape
        num_dimensions = 2
        heads = 8
        dim_heads = None
        dim_index = -1
        sum_axial_out = True
    """

    def __init__(self, dim, shape, num_dimensions = 2, heads = 8, dim_heads = None, dim_index = -1, sum_axial_out = True):
        """Construct a MultiHeadedAxialSelfAttention object."""
        super(MultiHeadedMedicalAxialSelfAttentionWrapper, self).__init__()
        self.shape = shape
        # dim should be the size of the dimension along which axial attention is applied.
        # shape is ignored here.
        # num dimensions is ignored here. this module assumes 3d tensors
        # heads is ignored. this module doesn't support it now.
        # dim_heads is ignored.

        if dim_index == -1:
            using_width = True
        elif dim_index == -2:
            using_width = False

        # use dim for kernel size, because it seems like its used
        self.forward_attention = AxialAttention(dim, dim, dim, num_dimensions, width=using_width)

    def forward(self, query_ignored, key, value_ignored, mask_ignored):
        """Wrapper for computing axial attention.

        Args:
            query (torch.Tensor): Query tensor ignored
            key (torch.Tensor): Key tensor (#batch, time2, original_size).
            value (torch.Tensor): Value tensor ignored.
            mask (torch.Tensor): Mask tensor ignored

        Returns:
            torch.Tensor: Output tensor (#batch, time1, original_size).

        """
        key_tensor = torch.unflatten(key, 2, self.shape)  # unflatten just for applying multihead
        b, t, c, h, w = key_tensor.shape
        key_tensor = key_tensor.contiguous().view(b * t, c, h, w)
        out = self.forward_attention(query_ignored, key_tensor, value_ignored, mask_ignored)
        out = out.contiguous().view(b, t, c, h, w)
        out = torch.flatten(out, start_dim=2)
        return out

###############################################################################
#                                    axial                                    #
###############################################################################

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, heads=1, groups=8,
                 stride=1, bias=False, width=False, with_position=False, with_gate=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        assert (self.group_planes >= 2)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width
        self.with_position = with_position
        self.with_gate = with_gate

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)

        if with_position:
            self.bn_similarity = nn.BatchNorm2d(groups * 3)
            self.bn_output = nn.BatchNorm1d(out_planes * 2)
        else:
            self.bn_similarity = nn.BatchNorm2d(groups)
            self.bn_output = nn.BatchNorm1d(out_planes * 1)

        # Position embedding
        if with_position:
            if with_gate:
                self.f_qr = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
                self.f_kr = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
                self.f_sve = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
                self.f_sv = nn.Parameter(torch.tensor(1.0),  requires_grad=False)

            self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
            query_index = torch.arange(kernel_size).unsqueeze(0)
            key_index = torch.arange(kernel_size).unsqueeze(1)
            relative_index = key_index - query_index + kernel_size - 1
            self.register_buffer('flatten_index', relative_index.view(-1))

        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, query_ignored, x, value_ignored, mask):
        """Compute ...

        Args:
            x (torch.Tensor): Query, key, and value tensors (#batch * time1, channels, height, width).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch * time1, channels, height, width).

        """
        if self.width:
            x = x.permute(0, 2, 3, 1)  # N * T, H, W, C
        else:
            x = x.permute(0, 3, 2, 1)  # N * T, W, H, C
        NT, W, H, C = x.shape
        x = x.contiguous().view(NT * W, H, C)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))

        # groups * group_planes * 2 = output planes * 2, which qkv has because
        # qkv_transform was defined to output that many channels. then q, k, and
        # v should be formed for each group. group_planes == output planes * 2.

        # group_planes * 2 = group_planes // 2 + group_planes // 2 +
        # group_planes as long as group_planes >= 2.

        q, k, v = torch.split(qkv.reshape(NT * W, self.groups, self.group_planes * 2, C),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        if self.with_position:
            all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2,
                                                                                           self.kernel_size,
                                                                                           self.kernel_size)

            q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                                [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)

            qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
            kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)

            if self.with_gate:
                qr = torch.mul(qr, self.f_qr)
                kr = torch.mul(kr, self.f_kr)

        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        if self.with_position:
            stacked_similarity = torch.cat([qk, qr, kr], dim=1)
            stacked_similarity = self.bn_similarity(stacked_similarity).view(NT * W, 3, self.groups, C, C).sum(dim=1)
            similarity = F.softmax(stacked_similarity, dim=3)
            sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
            sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

            if self.with_gate:
                sv = torch.mul(sv, self.f_sv)
                sve = torch.mul(sve, self.f_sve)

            stacked_output = torch.cat([sv, sve], dim=-1).view(NT * W, self.out_planes * 2, C)
            output = self.bn_output(stacked_output).view(NT, W, self.out_planes, 2, C).sum(dim=-2)
        else:
            stacked_similarity = self.bn_similarity(qk).reshape(NT * W, 1, self.groups, C, C).sum(dim=1).contiguous()
            similarity = F.softmax(stacked_similarity, dim=3)
            sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
            sv = sv.reshape(NT * W, self.out_planes * 1, C).contiguous()
            output = self.bn_output(sv).reshape(NT, W, self.out_planes, 1, C).sum(dim=-2).contiguous()

        if self.width:
            # Now: NT, H, W, C
            output = output.permute(0, 3, 2, 1) # Return to normal shape
        else:
            # Now: NT, W, H, C
            output = output.permute(0, 3, 1, 2) # Return to normal shape

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))

        if self.with_position:
            nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

###############################################################################
#                                     old                                     #
###############################################################################


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an MultiHeadedAttention object."""
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class LegacyRelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (old version).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.

    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, time2).

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor (#batch, time1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.

    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[
            :, :, :, : x.size(-1) // 2 + 1
        ]  # only keep the positions from 0 to time2

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, 2*time1-1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)
