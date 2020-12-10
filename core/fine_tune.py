# coding=UTF-8
import torch
import torch.nn as nn
import numpy as np

from .evaluate import eval_encoder_classifier
from .util import get_optimizer, get_linear_schedule_with_warmup
from .loss import contrastive_loss
from .model import LNClassifier
from .fine_tune_milti_view import update_s, multi_view_learn


def fine_tune_encoder(opt, encoder, graph_data, seeds):
    n_class = opt['n_class']
    d_es = graph_data.x[0].size(-1)
    classifier = LNClassifier(d_es, n_class)
    classifier.to(opt['device'])
    criterion = torch.nn.CrossEntropyLoss()
    sim_metric = opt['sim_metric']
    parameters = [
        {'params': [p for p in encoder.parameters() if p.requires_grad],
         'lr': opt['lr'],
         'weight_decay': opt['decay']},
        {'params': [p for p in classifier.parameters() if p.requires_grad]}]
    optimizer = get_optimizer(opt['optimizer'], parameters,
                              opt['lr'], opt['decay'])
    steps = opt['init_encoder_epoch']
    warm_steps = steps * 0.1
    scheduler = get_linear_schedule_with_warmup(optimizer, warm_steps, steps)
    eval_encoder_classifier(encoder, classifier, graph_data, seeds)

    seed = torch.cat(seeds, dim=0)
    seed_label = graph_data.y[seed]
    for i in range(1, steps+1):
        encoder.train()
        classifier.train()
        optimizer.zero_grad()
        es, ps = encoder(graph_data)
        logits = classifier(es[seed])
        loss = criterion(logits, seed_label)
        un_loss = contrastive_loss((es, ps), graph_data, sim_metric)
        loss += opt['un_weight'] * un_loss
        loss.backward()
        # nn.utils.clip_grad_norm_(encoder.parameters(), opt['max_grad_norm'])
        # nn.utils.clip_grad_norm_(classifier.parameters(), opt['max_grad_norm'])
        optimizer.step()
        scheduler.step()
        print('Pre-train--Step: %d, loss:%.4f, lr:%.5f' %
              (i, loss.item(), scheduler.get_last_lr()[0]))
        if i % 50 == 0:
            eval_encoder_classifier(encoder, classifier, graph_data, seeds)
    return encoder, classifier


def fine_tune_decoder(opt, encoder, decoder, classifier, graph_data, seeds,
                      dev_seeds=None):
    update_s(opt, encoder, decoder, classifier, graph_data, seeds, mv_iter=0)
    multi_view_learn(opt, encoder, decoder, classifier, graph_data, seeds)
