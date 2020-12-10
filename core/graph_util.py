# coding=UTF-8
import os
from itertools import product, combinations
import torch
from .torch_util import scatter_mul
from torch_geometric.utils import dense_to_sparse, remove_self_loops


def load_node_neighbor(adj, file_path):
    from os.path import join as pjoin
    if file_path and os.path.exists(pjoin(file_path, 'node_neighbor.pt')):
        neighbor = torch.load(pjoin(file_path, 'node_neighbor.pt'))
    else:
        adj = adj.to('cuda:8')
        adj = adj > 0
        neighbor = torch.mm(adj.float(), adj.float().t()) > 0
        neighbor, _ = dense_to_sparse(neighbor)
        neighbor = remove_self_loops(neighbor)[0]
        neighbor = neighbor.cpu()
        if file_path:
            torch.save(neighbor, pjoin(file_path, 'node_neighbor.pt'))
    return neighbor


def load_node_pair(adj, file_path, n_hop=10):
    from os.path import join as pjoin
    if file_path and os.path.exists(pjoin(file_path, 'pos_node_edge.pt')):
        pos_neighbor = torch.load(pjoin(file_path, 'pos_node_edge.pt'))
        neg_neighbor = torch.load(pjoin(file_path, 'neg_node_edge.pt'))
    else:
        adj = adj.to('cuda:8')
        adj = adj > 0
        neighbor = torch.mm(adj.float(), adj.float().t()) > 0
        size = neighbor.size(1)

        pos_neighbor = torch.zeros_like(neighbor)
        sort_weight, indices = torch.sum(neighbor.float(), dim=0).sort()
        min_count, max_count = int(size * 0.1), int(size * 0.9)
        min_pos, max_pos = sort_weight[min_count], sort_weight[max_count]
        min_count = (sort_weight < max(min_pos, 2)).sum()
        max_count = (sort_weight <= max_pos).sum()
        pos_select = indices[min_count:max_count]
        pos_neighbor[:, pos_select] = neighbor[:, pos_select]

        neg_neighbor = neighbor
        for i in range(1, n_hop):
            neg_neighbor = torch.mm(neg_neighbor.float(), neighbor.float()) > 0
        neg_neighbor = ~ neg_neighbor

        pos_neighbor, _ = dense_to_sparse(pos_neighbor)
        pos_neighbor = remove_self_loops(pos_neighbor)[0]
        neg_neighbor, _ = dense_to_sparse(neg_neighbor)
        pos_neighbor = pos_neighbor.cpu()
        neg_neighbor = neg_neighbor.cpu()
        if file_path:
            torch.save(pos_neighbor, pjoin(file_path, 'pos_node_edge.pt'))
            torch.save(neg_neighbor, pjoin(file_path, 'neg_node_edge.pt'))
    return pos_neighbor, neg_neighbor


def load_pattern_pair(adj, file_path, n_hop=10):
    if file_path and os.path.exists(file_path+'/neg_edge.pt'):
        pos_mask = torch.load(file_path + '/pos_edge.pt')
        neg_mask = torch.load(file_path + '/neg_edge.pt')
    else:
        adj = adj.to('cuda:8')
        adj = adj > 0
        size = adj.size(1)

        pos_mask = torch.zeros_like(adj)
        sort_weight, indices = torch.sum(adj.float(), dim=0).sort()
        min_count, max_count = int(size * 0.1), int(size * 0.9)
        min_pos, max_pos = sort_weight[min_count], sort_weight[max_count]
        min_count = (sort_weight < max(min_pos, 2)).sum()
        max_count = (sort_weight <= max_pos).sum()
        pos_select = indices[min_count:max_count]
        pos_mask[:, pos_select] = adj[:, pos_select]

        neg_mask = adj
        adj = adj.float()
        for i in range(1, n_hop):
            neg_mask = torch.mm(neg_mask.float(), adj.t()) > 0
            neg_mask = torch.mm(neg_mask.float(), adj) > 0
        neg_mask = ~ neg_mask

        pos_mask, _ = dense_to_sparse(pos_mask)
        neg_mask, _ = dense_to_sparse(neg_mask)
        pos_mask = pos_mask.cpu()
        neg_mask = neg_mask.cpu()
        if file_path:
            torch.save(pos_mask, file_path + '/pos_edge.pt')
            torch.save(neg_mask, file_path + '/neg_edge.pt')
    return pos_mask, neg_mask


def noisy_or(probs, adj):
    log_rev_probs = (1. - probs.clamp(max=1-1e-5)).log()
    log_rev_entity_probs = torch.mm(adj, log_rev_probs)
    entity_probs = 1. - log_rev_entity_probs.exp()
    return entity_probs


def get_cate_mask(seeds, total_size):
    cate_masks = []
    for seed in seeds:
        cate_mask = torch.zeros(total_size, device=seed.device)
        cate_mask.scatter_(0, seed, 1)
        cate_masks.append(cate_mask.bool())
    return cate_masks


def get_mask(seeds, total_size, expansions=None):
    cate_mask = get_cate_mask(seeds, total_size)
    new_seeds = [[seed] for seed in seeds]
    if expansions is not None:
        for step_expansions in expansions:
            for i, expansion in enumerate(step_expansions):
                cate_mask[i].scatter_(0, expansion, 1)
                new_seeds[i].append(expansion.cpu())
    mask = torch.stack(cate_mask, dim=0).sum(dim=0).bool()
    return mask, cate_mask, new_seeds


def get_cate_neighbors(starts, edge_index, mask, min_count=1):
    cate_valid = []
    for cate_mask in starts:
        cate_mask = cate_mask.unsqueeze(-1).float()
        cate_candidate = get_neighbors(cate_mask, edge_index,
                                       min_count=min_count)
        cate_candidate = cate_candidate.squeeze(-1)
        cate_candidate[mask] = 0
        cate_valid.append(cate_candidate)
    return cate_valid


def get_neighbors(starts, edge_index, do_scatter=False, length=-1,
                  min_count=1, return_count=False):
    ''' 计算输入节点的邻接节点
    '''
    if do_scatter:
        starts = torch.zeros((length, 1), device=starts.device,
                             dtype=starts.dtype).scatter_(0, starts, 1)
    ends = scatter_mul(starts, edge_index)
    ends[ends < min_count] = 0
    if not return_count:
        ends = (ends > 0).int()
    return ends


def get_k_hop_adjacency(adj, k, file_path, bi_graph=False):
    '''计算k-hop以内(含k)的邻接节点
    '''
    file_name = os.path.join(file_path, '%d_hop_neighbor.pt' % k)
    if file_path and os.path.exists(file_name):
        output, depth = torch.load(file_name)
    else:
        if k < 2:
            output, depth = dense_to_sparse(adj.long().cpu())
            return output, depth
        adj = adj.bool()
        neighbor = adj.float()
        output = adj.long()
        k_neighbor = neighbor
        for i in range(2, k+1):
            # find the long-tail nodes
            '''
            degrees, indices = k_neighbor.sum(dim=1).sort()
            long_tail_degree = min(2, degrees[int(0.9 * degrees.size(0))])
            long_tail_indices = indices[degrees <= long_tail_degree]
            '''
            if bi_graph:
                k_neighbor = torch.mm(k_neighbor, neighbor.t()).bool()
                k_neighbor = torch.mm(k_neighbor.float(), neighbor)
            else:
                k_neighbor = torch.mm(k_neighbor, neighbor.t())

            # only retain k-hop neighborhood for long-tail nodes
            '''
            long_tail = torch.zeros_like(k_neighbor)
            long_tail[long_tail_indices, :] = k_neighbor[long_tail_indices, :]
            '''

            # control the augmented links are less than existing links
            '''
            new_mask = long_tail.bool() & ~output.bool()
            counts = long_tail[new_mask]
            counts = counts.sort()[0]
            existing_count = output.bool().sum().long()
            add_count = min(counts.size(0), int(0.5 * existing_count.item()))
            min_count = max(1, counts[-add_count])
            '''

            # add augmented links with their depth
            k_adj = (k_neighbor > 1) & ~output.bool()
            output.masked_fill_(k_adj, i)
            k_neighbor = k_neighbor.bool().float()
        output = output.cpu()
        output, depth = dense_to_sparse(output)
        torch.save((output, depth), file_name)
    return output, depth


def group_mask(group_num, num_per_group, device=None):
    total = group_num * num_per_group
    pos_mask = torch.zeros((total, total), dtype=torch.bool, device=device)
    neg_mask = torch.zeros_like(pos_mask)

    for i in range(group_num):
        offset = i * num_per_group
        for j, k in combinations(range(num_per_group), 2):
            pos_mask[offset+j][offset+k] = 1
            pos_mask[offset+k][offset+j] = 1

    for i, j in combinations(range(group_num), 2):
        o_i = i * num_per_group
        o_j = j * num_per_group
        for k, l in product(range(num_per_group), range(num_per_group)):
            neg_mask[o_i + k][o_j + l] = 1
            neg_mask[o_j + l][o_i + k] = 1
    return pos_mask, neg_mask


def class_mask(class_num, device=None):
    neg_mask = torch.zeros((class_num, class_num),
                           dtype=torch.bool, device=device)

    for i, j in combinations(range(class_num), 2):
        neg_mask[i][j] = 1

    return neg_mask
