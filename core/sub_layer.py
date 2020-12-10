# coding=UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import ScaledDotProduct, FlattenScaledDotProduct
from .module import AdditiveMul, FlattenAdditiveMul
from torch_geometric.nn.inits import glorot, zeros


class MLP(nn.Module):
    def __init__(self, d_input, d_hidden, d_output, dropout=0., n_layer=2):
        super(MLP, self).__init__()
        self.d_input = d_input
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(d_input, d_hidden))
        self.dropout = dropout
        for i in range(n_layer - 2):
            self.fcs.append(nn.Linear(d_hidden, d_hidden))
        self.fcs.append(nn.Linear(d_hidden, d_output))
        self.reset()

    def reset(self):
        for fc in self.fcs:
            glorot(fc.weight)
            zeros(fc.bias)

    def forward(self, x):
        out = x
        for fc in self.fcs[:-1]:
            out = F.leaky_relu(fc(out))
        out = self.fcs[-1](out)
        return out


class GraphAttentionLayer(nn.Module):
    def __init__(self, d_in, d_in_k, d_in_v, n_head, d_head, d_v=None,
                 concat=True, bias=True, negative_slope=0.2,
                 attn_method='flatten_sdp', dropout=0.):
        super(GraphAttentionLayer, self).__init__()

        self.d_in = d_in
        self.n_head = n_head
        self.d_head = d_head
        self.d_v = d_v
        self.concat = concat
        self.bias = bias

        self.w_q = nn.Linear(d_in, n_head * d_head, bias=bias)
        self.w_k = nn.Linear(d_in_k, n_head * d_head, bias=bias)
        if d_v is not None:
            self.w_v = nn.Linear(d_in_v, n_head * d_v, bias=bias)

        self.attn_method = attn_method
        if attn_method == 'flatten_sdp':
            self.attention = FlattenScaledDotProduct(temperature=d_head ** 0.5,
                                                     dropout=dropout)
        elif attn_method == 'sdp':
            self.attention = ScaledDotProduct(temperature=d_head ** 0.5,
                                              dropout=dropout)
        elif attn_method == 'flatten_add':
            self.attention = FlattenAdditiveMul(n_head, d_head,
                                                dropout=dropout,
                                                negative_slope=negative_slope)
        elif attn_method == 'add':
            self.attention = AdditiveMul(n_head, d_head, dropout=dropout,
                                         negative_slope=negative_slope)
        else:
            raise ValueError('attn_method must in '
                             '["add","flatten_add","sdp","flatten_sdp"]')
        self.reset()

    def reset(self):
        glorot(self.w_q.weight)
        glorot(self.w_k.weight)
        if self.d_v is not None:
            glorot(self.w_v.weight)
        if self.bias:
            zeros(self.w_q.bias)
            zeros(self.w_k.bias)
            if self.d_v is not None:
                zeros(self.w_v.bias)

    def forward(self, q, k, v, **kwargs):
        d_head, n_head = self.d_head, self.n_head
        batchlize = q.dim() > 2
        if not batchlize:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        q = self.w_q(q)
        k = self.w_k(k)
        q = q.view(sz_b, len_q, n_head, d_head)
        k = k.view(sz_b, len_k, n_head, d_head)

        if self.d_v is not None:
            v = self.w_v(v)
            v = v.view(sz_b, len_v, n_head, self.d_v)
        else:
            v = v.view(sz_b, len_v, n_head, -1)

        if self.attn_method[-3:] == 'sdp':
            q = q.transpose(1, 2).view(-1, len_q, d_head)
            k = k.transpose(1, 2).view(-1, len_k, d_head)
            v = v.transpose(1, 2).view(-1, len_v, d_head)

        attn_score = self.attention(q, k, **kwargs)
        if self.attn_method[:8] == 'flatten_':
            attn_v = attn_score.unsqueeze(-1) * v
        else:
            attn_v = torch.bmm(attn_score, v)

        if self.attn_method[-3:] == 'sdp':
            attn_v = attn_v.view(sz_b, n_head, len_v, -1).transpose(1, 2)
            attn_v = attn_v.contiguous()

        if self.concat:
            attn_v = attn_v.view(sz_b, len_v, -1)
        else:
            attn_v = attn_v.mean(dim=-2)
        if not batchlize:
            attn_v = attn_v.squeeze(0)
            attn_score = attn_score.squeeze(0)
        return attn_v, attn_score


class EmbeddingEncodingLayer(nn.Module):
    def __init__(self, n_vocab, d_emb, padding_idx=None):
        super(EmbeddingEncodingLayer, self).__init__()
        self.encoding = nn.Embedding(n_vocab, d_emb, padding_idx=padding_idx)
        assert self.encoding.weight.requires_grad
        nn.init.orthogonal_(self.encoding.weight)

    def forward(self, x):
        return self.encoding(x)


class PositionEncodingLayer(nn.Module):
    def __init__(self, n_position, d_hid):
        super(PositionEncodingLayer, self).__init__()

        pos_table = self._get_sinusoid_encoding_table(n_position, d_hid)
        self.encoding = nn.Embedding.from_pretrained(pos_table, freeze=True)

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        def get_vec(pos):
            return [pos / np.power(1e4, 2*(j//2)/d_hid) for j in range(d_hid)]

        sinusoid_table = np.array([get_vec(pos) for pos in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table)

    def forward(self, x):
        return self.encoding(x)


class CombineLayer(nn.Module):
    def __init__(self, d_model, d_k, dropout=0.):
        super(CombineLayer, self).__init__()
        self.w_qs = nn.Linear(d_model,  d_k, bias=False)
        self.w_ks = nn.Linear(d_model,  d_k, bias=False)

        self.attn = ScaledDotProduct(temperature=np.power(d_k, 0.5),
                                     dropout=dropout)
        self.reset()

    def reset(self):
        glorot(self.w_qs.weight)
        glorot(self.w_ks.weight)

    def forward(self, q, k, v, mask=None):
        q = self.w_qs(q).unsqueeze(1)
        k = self.w_ks(k)

        attn_score = self.attn(q, k, mask)
        output = torch.bmm(attn_score, v).squeeze(1)
        return output
