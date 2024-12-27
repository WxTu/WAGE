from sklearn.utils import shuffle
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import KFold
from utils.misc import normalize_adj


def test_Profiling(feat, True_feat):
    for topK in [10, 20, 50]:
        avg_recall, avg_ndcg = RECALL_NDCG(feat, True_feat, topN=topK)
        print('topK: {}, recall: {}, ndcg: {}'.format(
            topK, avg_recall, avg_ndcg))


class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout):
        super(GCNLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(in_features, out_features).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)

    def forward(self, input, sp_adj, is_sp_fts=False):
        if is_sp_fts:
            h = torch.spmm(input, self.W)
        else:
            h = torch.mm(input, self.W)
        h_prime = torch.spmm(sp_adj, h)
        return F.elu(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN_eva(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, input_fts_sparse=True):
        """Dense version of GAT."""
        super(GCN_eva, self).__init__()
        self.dropout = dropout
        self.GCNlayer1 = GCNLayer(nfeat, nhid, dropout=dropout)
        self.GCNlayer2 = GCNLayer(nhid, nhid, dropout=dropout)
        self.input_fts_sparse = input_fts_sparse

        self.fc1 = nn.Linear(nhid, nclass)

    def forward(self, x, sp_adj):
        h1 = self.GCNlayer1(x, sp_adj, is_sp_fts=self.input_fts_sparse)
        h1 = F.dropout(h1, self.dropout, training=self.training)
        self.z = self.GCNlayer2(h1, sp_adj, is_sp_fts=False)

        h3 = F.log_softmax(self.fc1(self.z), dim=1)
        return h3


def RECALL_NDCG(estimated_fts, true_fts, topN=10):
    preds = np.argsort(-estimated_fts, axis=1)
    preds = preds[:, :topN]

    gt = [np.where(true_fts[i, :] != 0)[0] for i in range(true_fts.shape[0])]
    recall_list = []
    ndcg_list = []
    for i in range(preds.shape[0]):
        if len(gt[i]) != 0:
            if np.sum(estimated_fts[i, :]) != 0:
                recall = len(set(preds[i, :]) & set(
                    gt[i])) * 1.0 / len(set(gt[i]))
                recall_list.append(recall)

                intersec = np.array(list(set(preds[i, :]) & set(gt[i])))
                if len(intersec) > 0:
                    dcg = [np.where(preds[i, :] == ele)[0] for ele in intersec]
                    dcg = np.sum([1.0 / (np.log2(x + 1 + 1)) for x in dcg])
                    idcg = np.sum([1.0 / (np.log2(x + 1 + 1))
                                   for x in range(len(gt[i]))])
                    ndcg = dcg * 1.0 / idcg
                else:
                    ndcg = 0.0
                ndcg_list.append(ndcg)
            else:
                temp_preds = shuffle(np.arange(estimated_fts.shape[1]))[:topN]

                recall = len(set(temp_preds) & set(
                    gt[i])) * 1.0 / len(set(gt[i]))
                recall_list.append(recall)

                intersec = np.array(list(set(temp_preds) & set(gt[i])))
                if len(intersec) > 0:
                    dcg = [np.where(temp_preds == ele)[0] for ele in intersec]
                    dcg = np.sum([1.0 / (np.log2(x + 1 + 1)) for x in dcg])
                    idcg = np.sum([1.0 / (np.log2(x + 1 + 1))
                                   for x in range(len(gt[i]))])
                    ndcg = dcg * 1.0 / idcg
                else:
                    ndcg = 0.0
                ndcg_list.append(ndcg)

    avg_recall = np.mean(recall_list)
    avg_ndcg = np.mean(ndcg_list)

    return avg_recall, avg_ndcg
