# coding=UTF-8
import os

import numpy as np
import random
import argparse
import torch

from core.model import GBNEncoder, GBNDecoder
from core.sim_metric import EDSim, SDPSim, CosSim
from core.fine_tune import fine_tune_decoder, fine_tune_encoder
from core.dataloader import load_graph, load_seed


SIM_TABLE = {
    'ed': EDSim(),
    'sdp': SDPSim(),
    'cos': CosSim()
}


def regist_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/CoNLL')
    parser.add_argument('--input_model_file', type=str, default='')
    parser.add_argument('--output_model_file', type=str, default='')
    parser.add_argument('--method', type=str, default='multi_view')
    parser.add_argument('--sim_metric', type=str, default='cos',
                        help='sim metric function,choose from list  ['
                        '"ed"(euclidean distance),"sdp"(scaled dot product)'
                        ',"cos"(cosine)]')
    parser.add_argument('--n_iter', type=int, default=20)
    parser.add_argument('--min_match', type=int, default=1)
    parser.add_argument('--n_expansion', type=int, default=10)
    parser.add_argument('--k_hop', type=int, default=2)
    parser.add_argument('--un_weight', type=float, default=1e-2)
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--mean_updated', action='store_true')

    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--negative_slope', type=float, default=0.2)
    parser.add_argument('--bias', action='store_true')

    parser.add_argument('--device', type=int, default=7)
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--init_encoder_epoch', type=int, default=200)
    parser.add_argument('--init_decoder_epoch', type=int, default=200)
    parser.add_argument('--encoder_epoch', type=int, default=50)
    parser.add_argument('--decoder_epoch', type=int, default=50)
    parser.add_argument('--optimizer', default='adam', help='Optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--decay', type=float, default=1e-3,
                        help='Weight decay for optimization')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='gradient norm')
    parser.add_argument('--feature_type', default='glove')
    parser.add_argument('--feature_dim', type=int, default=50)
    parser.add_argument('--edge_feature_dim', type=int, default=5)

    args = parser.parse_args()
    return args


def print_settings(opt):
    print('==========================Parameters==============================')
    print('Dataset:\t\t', opt['dataset'])
    print('Input Model Path:\t', opt['input_model_file'])
    print('Output Model Path:\t', opt['output_model_file'])
    print('Fine-Tune Method:\t', opt['method'])
    print('Similarity Metric:\t', opt['sim_metric'])
    print('Bootstrapging Iter:\t', opt['n_iter'])
    print('Step Expansion:\t\t', opt['n_expansion'])
    print('Min Match Filter:\t', opt['min_match'])
    print('K Hop Neighbor:\t\t', opt['k_hop'])
    print('Unsuper Weight:\t\t', opt['un_weight'])
    print('Is Localized:\t\t', opt['local'])
    print('Mean Updated:\t\t', opt['mean_updated'])

    print('GNN Layers:\t\t', opt['n_layer'])
    print('Dropout:\t\t', opt['dropout'])
    print('Negative_slope:\t\t', opt['negative_slope'])
    print('Linear Bias:\t\t', opt['bias'])

    print('Random Seed:\t\t', opt['seed'])

    print('Init Encoder Epoch:\t', opt['init_encoder_epoch'])
    print('Init Decoder Epoch:\t', opt['init_decoder_epoch'])
    print('Encoder Epoch:\t\t', opt['encoder_epoch'])
    print('Decoder Epoch:\t\t', opt['decoder_epoch'])
    print('Optimizer:\t\t', opt['optimizer'])
    print('Learning Rate:\t\t', opt['lr'])
    print('Weight Decay:\t\t', opt['decay'])
    print('Max Grad Norm:\t\t', opt['max_grad_norm'])
    print('Feature Type:\t\t', opt['feature_type'])
    print('Feature Dim:\t\t', opt['feature_dim'])
    print('Edge Feature Dim:\t', opt['edge_feature_dim'])
    print('==================================================================')


if __name__ == '__main__':
    args = regist_parser()
    if args.seed:
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # cuda device setting
    if args.cpu and args.device is not None or not torch.cuda.is_available():
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda:%d' % args.device)
        if args.seed:
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

    opt = vars(args)
    if opt['local']:
        opt['k_hop'] = 1
    print_settings(opt)
    if opt['feature_type'] == 'bert':
        opt['feature_dim'] = 768
    device = opt['device']
    opt['sim_metric'] = SIM_TABLE[opt['sim_metric'].lower()].to(device)

    encoder = GBNEncoder(opt)
    if opt['input_model_file']:
        encoder.load_state_dict(torch.load(opt['input_model_file']+'.pth'))

    print('========================Do Fine-Tune==============================')
    pkl_path = 'graph_' + opt['feature_type'] + '.pkl'
    graph_data, graph = load_graph(opt, pkl_path, pyg=True)
    seed_file = opt['dataset'] + '/seeds.txt'
    seeds = load_seed(graph.node_s, seed_file)
    seeds = [torch.LongTensor(seed) for seed in seeds]
    opt['n_class'] = len(graph.node_s.itol)

    sim_metric = opt['sim_metric']
    decoder = GBNDecoder(opt, sim_metric)

    graph_data = graph_data.to(opt['device'])
    graph_data.x = (graph_data.x[0].to(opt['device']),
                    graph_data.x[1].to(opt['device']))
    seeds = [seed.to(opt['device']) for seed in seeds]
    encoder = encoder.to(opt['device'])
    decoder = decoder.to(opt['device'])

    encoder, classifier = fine_tune_encoder(opt, encoder, graph_data, seeds)
    fine_tune_decoder(opt, encoder, decoder, classifier, graph_data, seeds)
