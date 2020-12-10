# coding=UTF-8
import torch
from torch_scatter import scatter_sum


def scatter_mul(src, edge_index, edge_attr=None, dim=0):
    scatter_src = src.index_select(dim, edge_index[0])
    if edge_attr is not None:
        assert edge_index.size(1) == edge_attr.size(0)
        scatter_src = scatter_src * edge_attr.long()
    output = scatter_sum(scatter_src, edge_index[1], dim)
    return output


def set_parameter_decay(nets, decay_weight, non_decay_token=()):
    decay, non_decay = [], []
    for net in nets:
        for name, param in net.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias"):
                non_decay.append(param)
            else:
                flag_no_decay = False
                for t in non_decay_token:
                    if t in name:
                        flag_no_decay = True
                        break
                if flag_no_decay:
                    non_decay.append(param)
                else:
                    decay.append(param)
    return [{'params': non_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': decay_weight}]


def cosine_sim(a, b, dim=-1, eps=1e-8):
    """calculate the cosine similarity and avoid the zero-division
    """
    a_norm = a / (a.norm(dim=dim)[:, None]).clamp(min=eps)
    b_norm = b / (b.norm(dim=dim)[:, None]).clamp(min=eps)
    if len(a.shape) <= 2:
        sim = torch.mm(a_norm, b_norm.transpose(1, 0))
    else:
        sim = torch.einsum('ijk, lmk->iljm', (a_norm, b_norm))
    return sim


def inner_product(a, b, eps=1e-8):
    """calculate the inner product of two vectors
    """
    if len(a.shape) <= 2:
        sim = torch.mm(a, b.t())
    else:
        sim = torch.einsum('ijk, lmk->iljm', (a, b))
    return sim


def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    """
    copied from allenNLP
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (
                result.sum(dim=dim, keepdim=True) + tiny_value_of_dtype(result.dtype)
            )
        else:
            masked_vector = vector.masked_fill(~mask, min_value_of_dtype(vector.dtype))
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def masked_mean(vector: torch.Tensor,
                mask: torch.BoolTensor,
                dim: int,
                keepdim: bool = False) -> torch.Tensor:
    """
    copied from allenNLP
    """
    replaced_vector = vector.masked_fill(~mask, 0.0)

    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask, dim=dim, keepdim=keepdim)
    return value_sum / value_count.float().clamp(min=tiny_value_of_dtype(torch.float))


def info_value_of_dtype(dtype: torch.dtype):
    """
    copied from allenNLP
    """
    if dtype == torch.bool:
        raise TypeError("Does not support torch.bool")
    elif dtype.is_floating_point:
        return torch.finfo(dtype)
    else:
        return torch.iinfo(dtype)


def min_value_of_dtype(dtype: torch.dtype):
    """
    copied from allenNLP
    """
    return info_value_of_dtype(dtype).min


def max_value_of_dtype(dtype: torch.dtype):
    """
    copied from allenNLP
    """
    return info_value_of_dtype(dtype).max


def tiny_value_of_dtype(dtype: torch.dtype):
    """
    copied from allenNLP
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))
