# coding=utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .evaluate import evaluate, eval_encoder_classifier
from .util import predict_confidence
from .util import get_optimizer, get_linear_schedule_with_warmup


def multi_view_learn(opt, encoder, decoder, classifier, graph_data, seeds):
    for i in range(1, 6):
        update_t(opt, encoder, decoder, classifier, graph_data, seeds, i)
        update_s(opt, encoder, decoder, classifier, graph_data, seeds, i)


def update_s_data(encoder, classifier, graph_data, seeds):
    fake_target = torch.zeros(graph_data.m_y.size())
    fake_target = fake_target.to(graph_data.edge_index.device)
    with torch.no_grad():
        encoder.eval()
        classifier.eval()
        es, _ = encoder(graph_data)
        probs = F.softmax(classifier(es), dim=-1)
        confidence = predict_confidence(probs)
        '''
        idx_lb = torch.multinomial(probs, 1).squeeze(1)
        fake_target.zero_().scatter_(1, torch.unsqueeze(idx_lb, 1), 1.0)
        '''
        idx_lb = torch.argmax(probs, dim=-1, keepdim=True)
        fake_target.zero_().scatter_(1, idx_lb, 1.0)
    return fake_target, confidence


def update_s(opt, encoder, decoder, classifier, graph_data, seeds, mv_iter=0):
    fake_targets, confs = update_s_data(encoder, classifier, graph_data, seeds)
    edge_index = graph_data.node_edge_index
    encoder.eval()
    optimizer = get_optimizer(opt['optimizer'], decoder.parameters(),
                              opt['lr'] * (0.1 ** mv_iter),
                              opt['decay'] * (0.1 ** mv_iter))
    n_iter = opt['n_iter']
    n_epoch = opt['decoder_epoch'] if mv_iter else opt['init_decoder_epoch']
    warm_step = n_epoch * 0.1 if mv_iter == 0 else 0
    scheduler = get_linear_schedule_with_warmup(optimizer, warm_step, n_epoch,
                                                min_ratio=0.)
    for i in range(1, n_epoch+1):
        decoder.train()
        optimizer.zero_grad()
        es = encoder(graph_data)[0]
        es = es.detach()
        probs, selects, hxes = decoder.expand(es, edge_index, seeds, n_iter)
        loss = 0
        for ite, (iter_probs, iter_selects) in enumerate(zip(probs, selects)):
            score = np.exp(-ite/n_iter)
            select = torch.cat(iter_selects, dim=0)
            prob = torch.cat(iter_probs, dim=0)
            target = fake_targets[select]
            conf = confs[select]
            step_loss = - target * prob.log()
            loss += score * (conf * step_loss.sum(dim=-1)).mean()
        inner_loss = decoder.inner_loss(hxes)
        loss += inner_loss.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), opt['max_grad_norm'])
        optimizer.step()
        scheduler.step()
        print('Decoder Learn-step:%d, loss:%.5f, lr:%.6f' %
              (i, loss.item(), scheduler.get_last_lr()[0]))
        if i % 50 == 0:
            evaluate(encoder, decoder, graph_data, seeds, n_iter=n_iter)


def update_t_data(opt, encoder, decoder, graph_data, seeds):
    edge_index = graph_data.node_edge_index
    device = graph_data.x[0].device
    size = graph_data.x[0].size(0)
    target_t = torch.zeros((size, opt['n_class']),
                           dtype=torch.float, device=device)
    n_iter = opt['n_iter'] // 2 # avoid bad expansions
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        es, _ = encoder(graph_data)
        outputs, expansions, _ = decoder.expand(es, edge_index, seeds, n_iter)
        outputs = [torch.cat(output, dim=0) for output in outputs]
        outputs = torch.cat(outputs, dim=0)
        preds = []
        scores = []
        for i, cate_expansion in enumerate(expansions):
            score = 0.8 * np.exp(-i/n_iter)
            for j, expansion in enumerate(cate_expansion):
                preds.extend([j for _ in range(expansion.size(0))])
                scores.extend([score for _ in range(expansion.size(0))])
        preds = torch.from_numpy(np.array(preds)).to(dtype=torch.long,
                                                     device=device)
        scores = torch.from_numpy(np.array(scores)).to(dtype=torch.float,
                                                       device=device)
        expansions = [torch.cat(expansion, dim=0) for expansion in expansions]
        expansions = torch.cat(expansions, dim=0)
        confidence = torch.zeros(size, device=outputs.device)
        '''
        confidence[expansions] = predict_confidence(outputs)
        confidence[expansions] = confidence[expansions] * scores
        '''
        confidence[expansions] = scores
        target_t[expansions] = torch.scatter(target_t[expansions],
                                             1, preds.unsqueeze(1), 1.0)
        idx = torch.cat(seeds, dim=0)
        temp = torch.zeros(idx.size(0), target_t.size(1),
                           device=device).type_as(target_t)
        temp.scatter_(1, torch.unsqueeze(graph_data.y[idx], 1), 1.0)
        target_t[idx] = temp
        confidence[idx] = 1.
    return target_t, confidence, expansions


def update_t(opt, encoder, decoder, classifier, graph_data, seeds, mv_iter=0):
    f_tag, confs, fs = update_t_data(opt, encoder, decoder, graph_data, seeds)
    seed = torch.cat(seeds, dim=0)
    idx = torch.cat([seed, fs])
    encoder.train()
    classifier.train()
    parameters = [
        {'params': [p for p in encoder.parameters() if p.requires_grad]},
        {'params': [p for p in classifier.parameters() if p.requires_grad]}]
    optimizer = get_optimizer(opt['optimizer'], parameters,
                              opt['lr'] * (0.1 ** mv_iter),
                              opt['decay'] * (0.1 ** mv_iter))
    n_epoch = opt['encoder_epoch']
    warm_step = n_epoch * 0.1 if mv_iter == 0 else 0
    scheduler = get_linear_schedule_with_warmup(optimizer, warm_step, n_epoch,
                                                min_ratio=0.)
    for i in range(1, n_epoch+1):
        encoder.train()
        classifier.train()
        optimizer.zero_grad()
        es, ps = encoder(graph_data)
        probs = F.log_softmax(classifier(es), dim=-1)
        up_loss = torch.sum(-f_tag[idx] * probs[idx], dim=-1)
        loss = torch.mean(up_loss * confs[idx])
        loss.backward()
        optimizer.step()
        scheduler.step()
        print('Encoder learn-step:%d, loss:%.5f, lr:%.6f' %
              (i, loss.item(), scheduler.get_last_lr()[0]))
        if i % 50 == 0:
            eval_encoder_classifier(encoder, classifier, graph_data, seeds)
