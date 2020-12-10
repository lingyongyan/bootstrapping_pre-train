# coding=UTF-8
import torch
import torch.nn.functional as F
from inspect import isfunction


def multi_target_eval(preds, multi_target):
    results = torch.gather(multi_target, 1, torch.unsqueeze(preds, 1)).float()
    return results.view(-1)


def evaluate(encoder, decoder, graph_data, seeds,
             is_multi=True, idx_dev=None, n_iter=10):
    target, multi_target = graph_data.y, graph_data.m_y
    device = target.device
    if not isfunction(encoder) and encoder is not None:
        encoder.eval()
    decoder.eval()
    with torch.no_grad():
        if encoder is not None:
            es, _ = encoder(graph_data)
        else:
            es = graph_data.x[0]
        edge_index = graph_data.node_edge_index
        _, expansions, _ = decoder.expand(es, edge_index, seeds, n_iter)
        steps = [[len(exp) for exp in step_exp] for step_exp in expansions]
        expansions = [torch.cat(exp, dim=0) for exp in expansions]
        n_class = len(steps[0])

        print('===== evalutation=====')
        line = '\t'.join([str(i) for i in range(n_class)])
        print('>> step\t' + line + '\ttotal count\tp@n')
        total_count = 0
        full_count = 0
        for expansion in expansions:
            full_count += expansion.size(0)
        total_correct = .0
        preds = torch.full((full_count, ), -1, dtype=torch.long, device=device)
        full_start = 0
        for i, (expansion, step) in enumerate(zip(expansions, steps)):
            start = 0
            full_end = full_start + expansion.size(0)
            s_preds = torch.full((expansion.size(0), ), -1,
                                 dtype=torch.long, device=device)
            for j, length in enumerate(step):
                end = start + length
                i_end = full_start + length
                if end == start:
                    continue
                s_preds[start:end] = j
                preds[full_start:i_end] = j
                start = end
                full_start = i_end
            full_start = full_end
            if is_multi:
                s_tag = multi_target[expansion]
                s_correct = multi_target_eval(s_preds, s_tag).sum()
            else:
                s_tag = target[expansion]
                s_correct = s_preds.eq(s_tag).float().sum()
            total_correct += s_correct
            total_count += expansion.size(0)
            total_acc = total_correct / total_count
            s_temp = []
            for j in range(n_class):
                ss_select = s_preds == j
                ss_preds = s_preds[ss_select]
                ss_index = expansion[ss_select]
                if is_multi:
                    ss_tag = multi_target[ss_index]
                    ss_correct = multi_target_eval(ss_preds, ss_tag).sum()
                else:
                    ss_tag = target[ss_index]
                    ss_correct = ss_preds.eq(ss_tag).float().sum()
                ss_count = ss_preds.size(0)
                ss_acc = float(ss_correct) / max(ss_count, 1)
                s_temp.append('%d/%d(%.1f)' % (ss_correct, ss_count, ss_acc))
            s_line = '\t'.join(s_temp)
            print('>> %d\t%s\t%d\t%.4f' %
                  (i+1, s_line, total_count, total_acc))

        expansions = torch.cat(expansions, dim=0)
        results = [[] for _ in range(n_class)]
        for i in range(n_class):
            sub_expansions = preds == i
            s_preds = preds[sub_expansions]
            sub_index = expansions[sub_expansions]
            if is_multi:
                s_target = multi_target[sub_index]
            else:
                s_target = target[sub_index]
            for t in range(10, 210, 10):
                if s_preds.size(0) < t:
                    break
                if is_multi:
                    correct = multi_target_eval(s_preds[:t], s_target[:t]).sum()
                else:
                    correct = s_preds[:t].eq(s_target[:t]).float().sum()
                accuracy = correct / t
                results[i].append(accuracy.item())
        '''
        print('>>> details <<<')
        line = '\t'.join([str(i) for i in range(n_class)])
        print('>> N\t' + line + '\ttotal')
        # assert expansions.size(0) == expansions.unique().size(0)
        for s, t in enumerate(range(10, 210, 10)):
            line = '>> %d' % t
            is_total = True
            total = []
            for i in range(n_class):
                if len(results[i]) > s:
                    total.append(results[i][s])
                    line += '\t%.3f' % results[i][s]
                else:
                    is_total = False
                    line += '\t-'
            if is_total:
                line += '\t%.4f' % (sum(total) / len(total))
            else:
                line += '\t-'
            print(line)
        '''


def eval_encoder_classifier(encoder, classifier, graph_data, tr_seeds, sub_idx=None):
    encoder.eval()
    classifier.eval()
    yy, device = graph_data.y, graph_data.y.device
    mask = torch.ones_like(yy, dtype=torch.bool, device=device)
    if tr_seeds is not None:
        mask[torch.cat(tr_seeds, dim=0)] = 0
    with torch.no_grad():
        es, _ = encoder(graph_data)
        probs = F.softmax(classifier(es), dim=-1)
        preds = torch.max(probs, dim=1)[1]
        if sub_idx is not None:
            probs = probs[sub_idx]
            preds = preds[sub_idx]
            target = graph_data.y[sub_idx]
        else:
            probs = probs[mask]
            preds = preds[mask]
            target = graph_data.y[mask]
        results = [[] for _ in range(probs.size(1))]
        entropy = torch.sum(probs * probs.log(), dim=-1)
        # entropy = probs.max(dim=-1)[0]
        # entropy = entropy[mask]
        for i in range(probs.size(1)):
            selects = preds == i
            s_entropy = entropy[selects]
            s_preds = preds[selects]
            s_target = target[selects]
            sort = torch.argsort(-s_entropy)
            for t in range(20, 440, 20):
                if sort.size(0) < t:
                    break
                correct = s_preds[sort[:t]].eq(s_target[sort[:t]]).double()
                accuracy = correct.sum() / t
                results[i].append(accuracy.item())
        print('===== teacher eval =====')
        line = '\t'.join([str(i) for i in range(probs.size(1))])
        print('>>>\tstep\t' + line + '\ttotal')
        for s, t in enumerate(range(20, 440, 20)):
            line = '>>>\t%d' % t
            is_total = True
            total = []
            for i in range(probs.size(1)):
                if len(results[i]) > s:
                    total.append(results[i][s])
                    line += '\t%.3f' % results[i][s]
                else:
                    is_total = False
                    line += '\t-'
            if is_total:
                line += '\t%.4f' % (sum(total) / len(total))
            else:
                line += '\t-'
            print(line)

        correct = preds.eq(target).double()
        acc = correct.sum() / target.size(0)
        total_num = target.size(0)
        print('>>> accuracy of teacher [%d] is %f' % (total_num, acc.item()))

