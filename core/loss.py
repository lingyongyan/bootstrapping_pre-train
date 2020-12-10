# coding=utf-8
import torch
import random
import torch.nn as nn
import numpy as np

from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.utils import negative_sampling


def contrastive_loss(encoder_output, graph_data, sim_metric):
    es, ps = encoder_output
    e_size = graph_data.x[0].size(0)

    ee_pos = graph_data.node_pos_index
    ee_neg = _contrastive_sample(ee_pos.size(1), graph_data.node_neg_index)
    ep_pos = graph_data.edge_pos_index
    ep_neg = _contrastive_sample(ep_pos.size(1), graph_data.edge_neg_index)

    ep1, ep2 = es.index_select(0, ee_pos[0]), es.index_select(0, ee_pos[1])
    link_sim = sim_metric(ep1, ep2, flatten=True, method='exp')
    en1, en2 = es.index_select(0, ee_neg[0]), es.index_select(0, ee_neg[1])
    non_sim = sim_metric(en1, en2, flatten=True, method='exp')
    pp1, pp2 = es.index_select(0, ep_pos[0]), ps.index_select(0, ep_pos[1])
    pos_sim = sim_metric(pp1, pp2, flatten=True, method='exp')
    pn1, pn2 = es.index_select(0, ep_neg[0]), ps.index_select(0, ep_neg[1])
    neg_sim = sim_metric(pn1, pn2, flatten=True, method='exp')

    en_sum = scatter_sum(non_sim, ee_neg[0], dim=-1, dim_size=e_size)
    link_loss = link_sim / (link_sim + en_sum.index_select(0, ee_pos[0]))
    ep_sum = scatter_sum(neg_sim, ep_neg[0], dim=-1, dim_size=e_size)
    ep_loss = pos_sim / (pos_sim + ep_sum.index_select(0, ep_pos[0]))

    loss = torch.cat([-link_loss.log(), -ep_loss.log()], dim=-1).mean()
    return loss


def edge_mask_loss(encoder_output, graph_data, masked_indice, classifier):
    edge_index = graph_data.edge_index
    edge_index = edge_index[:, masked_indice]
    size = (graph_data.x[0].size(0), graph_data.x[1].size(0))
    neg_edge_index = _negative_sample(graph_data.edge_index, size,
                                      num_neg=masked_indice.size(0))
    es, ps = encoder_output
    criterion = nn.BCEWithLogitsLoss()
    pos_score = classifier(torch.cat([es[edge_index[0]], ps[edge_index[1]]], dim=-1))
    neg_score = classifier(torch.cat([es[neg_edge_index[0]], ps[neg_edge_index[1]]], dim=-1))
    loss = criterion(pos_score, torch.ones_like(pos_score)) + \
        criterion(neg_score, torch.zeros_like(neg_score))
    loss = loss / 2
    return loss


def _contrastive_sample(pos_size, neg_index):
    # limited by the GPU memory, we randomly sample some neg-neighbors
    neg_size = pos_size * 5
    indices = torch.randperm(neg_index.size(1), device=neg_index.device)[:neg_size]
    neg_sample = neg_index[:, indices]
    return neg_sample


def _negative_sample(edge_index, size, num_neg):
    # Handle '|V|^2 - |E| < |E|'.
    count = size[0] * size[1]
    num_neg = min(num_neg, count - edge_index.size(1))

    row, col = edge_index
    idx = row * size[1] + col

    alpha = 1 / (1 - 1.2 * (edge_index.size(1) / count))

    perm = sample(count, int(alpha * num_neg))
    mask = torch.from_numpy(np.isin(perm, idx.to('cpu'))).to(torch.bool)
    perm = perm[~mask][:num_neg].to(edge_index.device)
    row = perm // size[1]
    col = perm % size[1]
    neg_edge_index = torch.stack([row, col], dim=0)
    return neg_edge_index


def sample(high: int, size: int, device=None):
    size = min(high, size)
    return torch.tensor(random.sample(range(high), size), device=device)
