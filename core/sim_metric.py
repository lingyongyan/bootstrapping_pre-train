# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot
from torch_scatter import scatter_softmax

from .torch_util import cosine_sim, masked_softmax


def _post_process(score, flatten, method, **kwargs):
    if method == 'sigmoid':
        score = torch.sigmoid(score)
    elif method == 'exp':
        score = torch.exp(score.clamp(max=70))
    elif method == 'softmax':
        mask = kwargs['mask'] if 'mask' in kwargs else None
        index = kwargs['index'] if 'index' in kwargs else None
        dim = kwargs['dim'] if 'dim' in kwargs else -1
        if flatten:
            assert index is not None
            score = scatter_softmax(score, index, dim=dim)
        else:
            score = masked_softmax(score, mask=mask, dim=dim)
    else:
        assert method == 'origin'
    return score


class EDSim(nn.Module):
    def __init__(self,
                 d_in: int = -1,
                 d_k: int = -1,
                 paramize: bool = False):
        super(EDSim, self).__init__()

        self.paramize = paramize
        if self.paramize:
            assert d_in > 0
            assert d_k > 0
            self.d_in = d_in
            self.d_k = d_k
            self.fc = nn.Linear(d_in, d_k, bias=False)
        self.reset()

    def reset(self):
        if self.paramize:
            glorot(self.fc.weight)

    def forward(self, q, k, flatten=False, method='origin', **kwargs):
        if not flatten:
            n, d, m = q.size(0), q.size(1), k.size(0)
            q = q.unsqueeze(1).expand(n, m, d)
            k = k.unsqueeze(0).expand(n, m, d)
        distance = q - k
        if self.paramize:
            distance = self.fc(distance)
        distance = torch.norm(distance, p=2, dim=-1)
        score = - distance / 2

        score = _post_process(score, flatten, method, **kwargs)
        return score


class CosSim(nn.Module):
    def __init__(self,
                 d_in: int = -1,
                 d_k: int = -1,
                 paramize: bool = False):
        super(CosSim, self).__init__()

        self.paramize = paramize
        if self.paramize:
            assert d_in > 0
            assert d_k > 0
            self.fc = nn.Linear(d_in, d_k, bias=False)
        self.reset()

    def reset(self):
        if self.paramize:
            glorot(self.fc.weight)

    def forward(self, q, k, flatten=False, method='origin', **kwargs):
        if self.paramize:
            q, k = self.fc(q), self.fc(k)
        if flatten:
            score = F.cosine_similarity(q, k, dim=-1)
        else:
            score = cosine_sim(q, k)

        score = _post_process(score, flatten, method, **kwargs)
        return score


class SDPSim(nn.Module):
    '''Scaled dot product similarity'''
    def __init__(self,
                 d_in: int = -1,
                 d_k: int = -1,
                 paramize: bool = False):
        super(SDPSim, self).__init__()

        self.paramize = paramize
        if self.paramize:
            assert d_in > 0
            assert d_k > 0
            self.fc = nn.Linear(d_in, d_k, bias=False)
        self.reset()

    def reset(self):
        if self.paramize:
            glorot(self.fc.weight)

    def forward(self, q, k, flatten=False, method='origin', **kwargs):
        if self.paramize:
            q, k = self.fc(q), self.fc(k)
        temperature = np.sqrt(q.size(1))
        if flatten:
            score = torch.einsum('ij, ij->i', q, k)
        else:
            score = torch.einsum('ik, jk->ij', q, k)
        score = score / temperature
        score = _post_process(score, flatten, method, **kwargs)
        return score
