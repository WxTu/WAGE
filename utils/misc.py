import copy
import numpy as np
import random
import torch
import dgl
from os import path as osp
import os
from sklearn.utils import shuffle
import scipy.sparse as sp
import yaml


def set_random_seed(seed):
    torch.manual_seed(seed)
    dgl.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def pyg2dgl(pygdata):
    x = pygdata.x
    edge_index = pygdata.edge_index
    src_ids = edge_index[0]
    dst_ids = edge_index[1]
    label = pygdata.y
    dgl_graph = dgl.graph((src_ids, dst_ids), num_nodes=x.shape[0])
    dgl_graph = dgl.add_self_loop(dgl_graph)
    dgl_graph.ndata['feat'] = x
    dgl_graph.ndata['label'] = label

    graph = copy.deepcopy(dgl_graph)
    return graph


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def data_split(args, num_nodes):
    shuffled_nodes = shuffle(np.arange(num_nodes), random_state=args.split_seed)
    train_fts_idx = torch.from_numpy(shuffled_nodes[:int(args.train_fts_ratio * num_nodes)]).long()
    vali_fts_idx = torch.from_numpy(
        shuffled_nodes[
        int(args.train_fts_ratio * num_nodes):int((args.train_fts_ratio + 0.1) * num_nodes)]).long()
    test_fts_idx = torch.from_numpy(shuffled_nodes[int((args.train_fts_ratio + 0.1) * num_nodes):]).long()
    vali_test_fts_idx = torch.from_numpy(shuffled_nodes[int(args.train_fts_ratio * num_nodes):]).long()
    print("Dataset loading done!")
    return train_fts_idx, vali_fts_idx, test_fts_idx, vali_test_fts_idx


def adj_normalized_cuda(adj, type='sys'):
    row_sum = torch.sum(adj, dim=1)
    row_sum = (row_sum == 0) * 1 + row_sum
    if type == 'sys':
        d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return d_mat_inv_sqrt.mm(adj).mm(d_mat_inv_sqrt)
    else:
        d_inv = torch.pow(row_sum, -1).flatten()
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diag(d_inv)
        return d_mat_inv.mm(adj)


def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def zero_filling(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(0), x.size(1)),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[drop_mask] = 0
    return x, drop_mask


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)
    configs = configs[args.dataset]
    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args


def compute_high_order_adj(dense_adj, max_order):
    adj_matrices = [dense_adj]
    one = torch.ones_like(dense_adj)
    zero = torch.zeros_like(dense_adj)

    for order in range(1, max_order):
        adj_current = torch.mm(dense_adj, adj_matrices[-1])
        adj_current -= torch.diag_embed(torch.diag(adj_current))
        adj_current = torch.where(adj_current > 0, one, zero)

        for prev_order in range(0, order):
            adj_current -= adj_matrices[prev_order]
            adj_current[adj_current < 0] = 0

        adj_matrices.append(adj_current)
    return adj_matrices
