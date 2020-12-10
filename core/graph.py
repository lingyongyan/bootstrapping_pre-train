# coding=UTF-8
import torch
import numpy as np
import math
import pickle

PAD = '<pad>'


class BiGraph(object):
    def __init__(self, node_s, node_t, edges):
        self.node_s = node_s
        self.node_t = node_t
        self.edges = edges

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        return obj

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


class LabeledSet(object):
    def __init__(self, itos=None, stoi=None, itol=None, ltoi=None, labels=None, multi=False):
        self.itos = itos if itos is not None else []
        self.stoi = stoi if stoi is not None else {}
        self.itol = itol if itol is not None else []
        self.ltoi = ltoi if ltoi is not None else {}
        self.labels = labels if labels is not None else []
        self.multi = multi

    def add(self, item, label='', **kwargs):
        label_id = self._add_label(label)
        item = self._pre_add(item)
        self._add_item(item, label_id, **kwargs)

    def _pre_add(self, item):
        return item

    def _add_label(self, label):
        if label:
            if not isinstance(label, str):
                label = str(label)
            if label not in self.ltoi:
                self.ltoi[label] = len(self.itol)
                self.itol.append(label)
                return len(self.itol) - 1
            else:
                return self.ltoi.get(label, None)
        else:
            return None

    def _add_item(self, item, label_id, **kwargs):
        if item not in self.stoi:
            self.stoi[item] = len(self.itos)
            self.itos.append(item)
            if label_id is not None:
                if self.multi:
                    label_id = [label_id]
                self.labels.append(label_id)
        elif self.multi:
            node_id = self.stoi[item]
            self.labels[node_id].append(label_id)

    def load_label_from_file(self, file_name, label_col, split='\t'):
        with open(file_name, 'r') as f:
            for line in f:
                items = line.strip().split(split)
                assert label_col < len(items)
                label = items[label_col]
                self._add_label(label)


class NodeSet(LabeledSet):
    def __init__(self, itos=None, stoi=None, itol=None, ltoi=None, labels=None, features=None, with_padding=False, multi=False, **kwargs):
        super(NodeSet, self).__init__(itos=itos, stoi=stoi, itol=itol, ltoi=ltoi, labels=labels, multi=multi)
        self.features = features

        for key, value in kwargs.items():
            setattr(self, key, value)

        if with_padding:
            self.add(PAD, 'pad')

    def _pre_add(self, node):
        if not isinstance(node, str):
            node = str(node)
        return node

    def __len__(self):
        return len(self.itos)

    def load_node_from_file(self, file_name, node_col, label_col=None,
                            multi_label=False, split='\t'):
        with open(file_name, 'r') as f:
            for line in f:
                items = line.strip().split(split)
                assert node_col < len(items)
                node = items[node_col]
                label_str = items[label_col] if label_col else ''
                if label_str and multi_label:
                    labels = items[label_col].split(' ')
                    for label in labels:
                        label = label.strip()
                        self.add(node, label=label)
                else:
                    self.add(node, label=label_str)

    def load_feature_from_file(self, file_name, node_col,
                               feature_settings, sparse=False, split='\t'):
        if sparse:
            feature_vocab, feature_col = feature_settings
            f_size = len(feature_vocab)
        else:
            f_size, feature_col = feature_settings

        if self.features is None:
            self.features = np.zeros((len(self.itos), f_size))

        with open(file_name, 'r') as f:
            for line in f:
                items = line.strip().split(split)
                node, feature = items[node_col], items[feature_col]
                node_id = self.stoi.get(node, -1)
                if node_id == -1:
                    continue
                if sparse:
                    for pair in feature.strip().split(' '):
                        col, w = pair.split(':')
                        col, w = feature_vocab.stoi.get(f, -1), float(w)
                        self.feature[node_id, col] = w
                else:
                    feature_split = feature.strip().split(' ')
                    feature_float = [float(w) for w in feature_split]
                    self.features[node_id] = feature_float


class EdgeSet(LabeledSet):
    def __init__(self, itos=None, stoi=None, itol=None, ltoi=None, labels=None, features=None, weights=None, directed=False, **kwargs):
        super(EdgeSet, self).__init__(itos=itos, stoi=stoi, itol=itol, ltoi=ltoi, labels=labels)
        self.features = features
        self.weights = weights if weights is not None else []
        self.directed = directed

        for key, value in kwargs.items():
            setattr(self, key, value)

    def _add_item(self, item, label_id, weight=None):
        if item not in self.stoi:
            self.stoi[item] = len(self.itos)
            self.itos.append(item)
            if label_id:
                self.labels.append(label_id)
            if weight:
                self.weights.append(weight)

    def load_edge_from_file(self, file_name, node_s, col_s, node_t, col_t, col_weight=None, split='\t'):
        with open(file_name, 'r') as f:
            for line in f:
                items = line.strip().split(split)
                source, target = items[col_s], items[col_t]
                w = float(items[col_weight]) if col_weight is not None and items[col_weight] else 1
                s = node_s.stoi.get(source, -1)
                t = node_t.stoi.get(target, -1)
                if w > 0 and s > -1 and t > -1:
                    self.add((s, t), weight=w)
