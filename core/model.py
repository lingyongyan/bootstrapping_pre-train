# coding=UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_mean

from .module import Expander
from .layer import GNNConv, MemoryLayer
from .sub_layer import MLP
from .graph_util import get_cate_mask, get_cate_neighbors

n_depth = 2 + 1
n_edge = 3 + 1


class model_check(nn.Module):
    def save(self, optimizer, filename):
        params = {
            'model': self.state_dict(),
            'optim': optimizer.state_dict()
        }
        try:
            print('print model to path:%s' % filename)
            torch.save(params, filename)
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, optimizer, filename, device):
        try:
            print('load model from path:%s' % filename)
            checkpoint = torch.load(filename, map_location=device)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        return optimizer


class GBNEncoder(nn.Module):
    def __init__(self, opt, JK="last"):
        super(GBNEncoder, self).__init__()
        self.n_layer = opt['n_layer']
        self.d_feature = opt['feature_dim']
        self.d_edge = opt['edge_feature_dim']
        self.dropout = opt['dropout']
        self.JK = JK

        if self.n_layer < 1:
            raise ValueError("Number of GNN layers must be greater than 0.")

        # List of GNNLayers
        self.node_nns = nn.ModuleList()
        self.edge_nns = nn.ModuleList()
        for layer in range(self.n_layer):
            self.node_nns.append(GNNConv(self.d_feature, self.d_edge,
                                         dropout=opt['dropout'],
                                         negative_slope=opt['negative_slope'],
                                         bias=opt['bias'],
                                         global_sighted=not opt['local'],
                                         flow='target_to_source'))
            self.edge_nns.append(GNNConv(self.d_feature, self.d_edge,
                                         dropout=opt['dropout'],
                                         negative_slope=opt['negative_slope'],
                                         bias=opt['bias'],
                                         global_sighted=not opt['local'],
                                         flow='source_to_target'))

        self.norms = torch.nn.ModuleList()
        for layer in range(self.n_layer):
            self.norms.append(nn.BatchNorm1d(self.d_feature))

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        h_list = [x]
        for layer in range(self.n_layer):
            h_i = self.node_nns[layer](h_list[layer], edge_index, edge_attr)
            h_j = self.edge_nns[layer](h_list[layer], edge_index, edge_attr)

            h = torch.cat([h_i, h_j], dim=0)
            h = self.norms[layer](h)
            h_i, h_j = h[:h_i.size(0)], h[h_i.size(0):]

            if layer < self.n_layer - 1:
                h_i = F.relu(h_i)
                h_j = F.relu(h_j)
            h_i = F.dropout(h_i, self.dropout, training=self.training)
            h_j = F.dropout(h_j, self.dropout, training=self.training)
            h_list.append([h_i, h_j])

        # Different implementations of Jk-concat
        if self.JK == "concat":
            h_0 = torch.cat([h[0] for h in h_list[1:]], dim=1)
            h_1 = torch.cat([h[1] for h in h_list[1:]], dim=1)
            output = (h_0, h_1)
        elif self.JK == "last":
            output = h_list[-1]
        elif self.JK == "max":
            h0 = torch.stack([h[0] for h in h_list[1:]], dim=0)
            h0 = torch.max(h0, dim=0)[0]
            h1 = torch.stack([h[1] for h in h_list[1:]], dim=0)
            h1 = torch.max(h1, dim=0)[0]
            output = (h0, h1)
        elif self.JK == "sum":
            h0 = torch.stack([h[0] for h in h_list[1:]], dim=0)
            h0 = torch.sum(h0, dim=0)
            h1 = torch.stack([h[1] for h in h_list[1:]], dim=0)
            h1 = torch.sum(h1, dim=0)
            output = (h0, h1)

        return output


class GBNDecoder(nn.Module, Expander):
    def __init__(self, opt, sim_metric):
        nn.Module.__init__(self)
        Expander.__init__(self, opt)
        self.memory_layer = MemoryLayer(opt['feature_dim'],
                                        attented=not opt['mean_updated'])
        self.sim_metric = sim_metric
        self.min_match = opt['min_match']

    def expand(self, es, edge_index, seeds, n_iter):
        outputs = []
        expansions = []
        hxes = []
        if isinstance(seeds, list):
            n_class = len(seeds)
        else:
            n_class = 1
            seeds = [seeds]
        seed_all = torch.cat(seeds, dim=0)
        known_mask = torch.zeros(es.size(0)).to(es.device, torch.bool)
        known_mask.scatter_(0, seed_all, 1)
        last_expansion = seeds

        cate_masks = get_cate_mask(seeds, es.size(0))
        hx = None

        for i in range(n_iter):
            if torch.sum((known_mask == 0)).float() == 0:
                break
            hx = self.one_step(es, last_expansion, hx)

            cate_valid = get_cate_neighbors(cate_masks, edge_index, known_mask,
                                            min_count=self.min_match)
            cate_valid = torch.stack(cate_valid, dim=0).bool().detach().t()
            scores = self.sim_metric(es, hx)
            cate_expansions, cate_probs = self._select(scores, cate_valid)
            last_expansion = cate_expansions
            known_mask.scatter_(0, torch.cat(last_expansion).view(-1), 1)
            for j in range(n_class):
                cate_masks[j].scatter_(0, cate_expansions[j], 1)

            outputs.append(cate_probs)
            expansions.append(cate_expansions)
            hxes.append(hx)
        return outputs, expansions, hxes

    def siamese_learn(self, es, edge_index, seed1, seed2, n_iter):
        output1 = self.expand(es, edge_index, seed1, n_iter, exclusive=True)
        output2 = self.expand(es, edge_index, seed2, n_iter, exclusive=True)
        return output1, output2

    def one_step(self, es, seeds, hx=None):
        inp, mask = self._last_expansion(es, seeds)
        hx = self.memory_layer(inp, hx, mask)
        return hx

    def _last_expansion(self, es, seeds):
        n_class, d_feature, device = len(seeds), es.size(-1), es.device
        mask = torch.zeros([n_class, self.n_expansion],
                           dtype=torch.float, device=device)
        inputs = torch.zeros([n_class, self.n_expansion, d_feature],
                             dtype=torch.float, device=device)
        for i, seed in enumerate(seeds):
            if seed.nelement() > 0:
                step = seed.size(0)
                inputs[i, :step] = es[seed]
                mask[i, :step] = 1
        return inputs, mask

    def inner_loss(self, hxes):
        criterion = nn.BCEWithLogitsLoss()
        hx_indice = torch.triu_indices(hxes[0].size(0), hxes[0].size(0), 1)
        losses = []
        for i, hx in enumerate(hxes):
            sim = self.sim_metric(hx, hx)
            sim = sim[hx_indice[0], hx_indice[1]]
            loss = criterion(sim, torch.zeros_like(sim))
            losses.append(loss)
        losses = torch.stack(losses, dim=0)
        return losses


class LNClassifier(nn.Module):
    def __init__(self, d_feature, n_class):
        super(LNClassifier, self).__init__()
        self.d_feature, self.n_class = d_feature, n_class
        self.fc = MLP(d_feature, d_feature // 2, n_class)

    def forward(self, x):
        out = self.fc(x)
        return out


class NNClassifier(nn.Module):
    def __init__(self, sim_metric):
        super(NNClassifier, self).__init__()
        self.sim_metric = sim_metric

    def forward(self, x, y, y_label, mask=None):
        cluster = scatter_mean(y, y_label, dim=0)
        probs = self.sim_metric(x, cluster, method='softmax', mask=mask)
        return probs
