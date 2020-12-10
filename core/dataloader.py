# coding=UTF-8
import os

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops

from .graph import BiGraph, NodeSet, EdgeSet
from .graph_util import load_node_neighbor, get_k_hop_adjacency
from .graph_util import load_node_pair, load_pattern_pair


def load_graph(opt, graph_pickle_path, pyg=True):
    data_dir = opt['dataset']
    feature_type = opt['feature_type']
    feature_dim = opt['feature_dim']
    k_hop = opt['k_hop']
    graph = _load_file(data_dir, graph_pickle_path, feature_type, feature_dim)
    if pyg:
        data = convert_graph_to_pyg(graph, k_hop, data_dir)
        print('1-hop link counts:', data.origin_edge.size())
        if k_hop > 1:
            print('%d-hop link counts:' % k_hop, data.edge_index.size())
        else:
            assert data.origin_edge.size(1) == data.edge_index.size(1)
        return data, graph
    return graph


def _load_file(dir, graph_pickle_path, feature_type, feature_dim):
    graph_pickle_path = os.path.join(dir, graph_pickle_path)
    if os.path.exists(graph_pickle_path):
        return BiGraph.load(graph_pickle_path)

    entities = NodeSet(multi=True)
    patterns = NodeSet()
    edges = EdgeSet()

    # load entities
    entity_label_path = os.path.join(dir, 'entity_labels.txt')
    entities.load_node_from_file(entity_label_path, 0, 1, multi_label=True)
    len_ent = len(entities)
    if feature_type == 'random':
        features = torch.randn(len_ent, feature_dim, requires_grad=True)
        entities.features = features
    elif feature_type == 'uniform':
        features = torch.full((len_ent, feature_dim), 0.02, requires_grad=True)
        entities.features = features
    else:
        e_feature_path = os.path.join(dir, 'e_feature_'+feature_type+'.txt')
        entities.load_feature_from_file(e_feature_path, 0, (feature_dim, 1))

    # load patterns
    pattern_path = os.path.join(dir, 'pattern_labels.txt')
    plabel_vocab_path = os.path.join(dir, 'pattern_label_vocab.txt')
    patterns.load_label_from_file(plabel_vocab_path, 0)
    patterns.load_node_from_file(pattern_path, 0, 1)
    len_pat = len(patterns)
    if feature_type == 'random':
        features = torch.randn(len_pat, feature_dim, requires_grad=True)
        patterns.features = features
    elif feature_type == 'uniform':
        features = torch.full((len_pat, feature_dim), 0.02, requires_grad=True)
        patterns.features = features
    else:
        p_feature_path = os.path.join(dir, 'p_feature_'+feature_type+'.txt')
        patterns.load_feature_from_file(p_feature_path, 0, (feature_dim, 1))

    # load edges
    edge_path = os.path.join(dir, 'links.txt')
    edges.load_edge_from_file(edge_path, entities, 0, patterns, 1)

    graph = BiGraph(entities, patterns, edges)
    graph.save(graph_pickle_path)
    return graph


def convert_graph_to_pyg(graph, k_hop, path):
    graph_file_name = os.path.join(path, 'graph_data_%d_hop.pt' % k_hop)
    if os.path.exists(graph_file_name):
        graph_data = torch.load(graph_file_name)
        return graph_data
    x_i = torch.tensor(graph.node_s.features, dtype=torch.float)
    x_j = torch.tensor(graph.node_t.features, dtype=torch.float)
    x = (x_i, x_j)
    y = [ls[0] for ls in graph.node_s.labels]
    y = torch.tensor(y, dtype=torch.long)
    label_size = len(graph.node_s.itol)
    m_y = torch.zeros((y.size(0), label_size), dtype=torch.long)
    for n, label in enumerate(graph.node_s.labels):
        m_y[n, label] = 1

    edge_index = torch.tensor(np.array(graph.edges.itos).T, dtype=torch.long)
    adj = torch.zeros((len(x_i), len(x_j)),
                      dtype=torch.float, device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = 1

    node_edge_index = load_node_neighbor(adj, path)
    # node_edge_index = remove_self_loops(node_edge_index)[0]
    node_pos_index, node_neg_index = load_node_pair(adj, path, n_hop=12)
    # node_pos_index = remove_self_loops(node_pos_index)[0]
    edge_pos_index, edge_neg_index = load_pattern_pair(adj, path, n_hop=20)

    new_edge_index, edge_depth = get_k_hop_adjacency(adj, k_hop, path, True)
    pat_label = torch.tensor(np.array(graph.node_t.labels), dtype=torch.long)
    edge_label = pat_label.index_select(0, new_edge_index[1])
    edge_attr = torch.stack([edge_depth, edge_label], dim=1)

    max_len = int(2e7)
    if edge_neg_index.size(1) > max_len:
        ll = np.random.choice(range(edge_neg_index.size(1)),
                              size=max_len, replace=False)
        edge_neg_index = edge_neg_index[:, ll]
    if node_neg_index.size(1) > max_len:
        ll = np.random.choice(range(node_neg_index.size(1)),
                              size=max_len, replace=False)
        node_neg_index = node_neg_index[:, ll]

    graph_data = Data(x=x, y=y, m_y=m_y,
                      edge_index=new_edge_index,
                      edge_attr=edge_attr,
                      origin_edge=edge_index,
                      node_edge_index=node_edge_index,
                      node_pos_index=node_pos_index,
                      node_neg_index=node_neg_index,
                      edge_pos_index=edge_pos_index,
                      edge_neg_index=edge_neg_index)
    torch.save(graph_data, graph_file_name)
    return graph_data


def load_seed(node_set, *seed_files):
    seed_dict = {}
    for seed_file in seed_files:
        with open(seed_file, 'r') as fi:
            for line in fi:
                node, lbl = line.strip().split('\t')
                ll = node_set.ltoi[lbl]
                if ll not in seed_dict:
                    seed_dict[ll] = []
                seed_dict[ll].append(node_set.stoi[node])
    max_seed_id = max(seed_dict.keys())
    assert max_seed_id == len(seed_dict) - 1
    seeds = [[] for _ in range(len(seed_dict))]
    for i in range(len(seed_dict)):
        seeds[i].extend(seed_dict[i])
    return seeds
