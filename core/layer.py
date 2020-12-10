# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import GRUCell

from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

from .sub_layer import GraphAttentionLayer, CombineLayer
from .sub_layer import MLP, EmbeddingEncodingLayer
from .torch_util import masked_mean

n_depth = 2 + 1
n_edge = 3 + 1


class GNNConv(MessagePassing):
    def __init__(self, d_in, d_attr, dropout=0.0,
                 negative_slope=0.2, eps=0, train_eps=True, bias=True,
                 global_sighted=True, concat=True, aggr='add',
                 flow='source_to_target'):
        super(GNNConv, self).__init__(aggr=aggr, flow=flow)
        n_head = 1
        d_head = d_in // n_head
        self.negative_slope = negative_slope
        self.bias = bias
        self.global_sighted = global_sighted
        if self.global_sighted:
            self.depth_encoder = EmbeddingEncodingLayer(n_depth, d_attr)
            self.edge_encoder = EmbeddingEncodingLayer(n_edge, d_attr)
        else:
            self.edge_encoder = None
            self.depth_encoder = None
        d_k = d_in + 2 * d_attr if self.global_sighted else d_in
        self.attn_layer = GraphAttentionLayer(d_in, d_k, d_in,
                                              n_head, d_head, d_head,
                                              concat=concat, bias=bias,
                                              negative_slope=negative_slope,
                                              attn_method='flatten_sdp',
                                              dropout=dropout)

        self.w_res = nn.Linear(d_in, d_in, bias=bias)
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.dropout = nn.Dropout(dropout)
        self.reset()

    def reset(self):
        self.eps.data.fill_(self.initial_eps)
        glorot(self.w_res.weight)
        if self.bias:
            zeros(self.w_res.bias)

    def forward(self, x, edge_index, edge_attr):
        size = (x[0].size(0), x[1].size(0))
        out = self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr)
        return out

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        q, v = x_i, x_j
        if self.global_sighted:
            depth_embedding = self.depth_encoder(edge_attr[:, 0])
            edge_embedding = self.edge_encoder(edge_attr[:, 1])
            k = torch.cat([x_j, depth_embedding, edge_embedding], dim=-1)
        else:
            k = x_j
        out, _ = self.attn_layer(q, k, v, index=edge_index_i, size=size_i)
        return out

    def update(self, aggr_out, x):
        res = x[0] if self.flow == 'target_to_source' else x[1]
        res = self.w_res(res)
        out = (1 + self.eps) * res + aggr_out
        return out


class MemoryLayer(nn.Module):
    def __init__(self, dim, attented=True, dropout=0.):
        super(MemoryLayer, self).__init__()
        self.combine_layer = CombineLayer(dim, dim//2, dropout=dropout)
        self.memory_cell = GRUCell(dim, dim)
        self.attented = attented

    def forward(self, inp, hx, mask=None):
        if self.attented:
            if hx is None:
                inp = inp.mean(dim=1)
            else:
                inp = self.combine_layer(hx, inp, inp, mask=mask)
        else:
            inp = masked_mean(inp, mask.unsqueeze(-1).bool(), dim=1)
        hx = self.memory_cell(inp, hx)
        return hx
