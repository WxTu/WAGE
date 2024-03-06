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


def test_AX(gene_data, labels_of_gene, adj):
    train_fts_ratio = 0.4 * 1.0
    print('begining test_AX......')
    is_cuda = torch.cuda.is_available()
    n_nodes = adj.shape[0]
    indices = np.where(adj != 0)
    rows = indices[0]
    cols = indices[1]
    adj = sp.coo_matrix(
        (np.ones(shape=len(rows)), (rows, cols)), shape=[n_nodes, n_nodes])
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    indices = torch.LongTensor(
        np.int64(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0)))
    values = torch.FloatTensor(adj.tocoo().data)
    adj = torch.sparse.FloatTensor(indices, values, torch.Size(adj.shape))
    labels_of_gene = torch.LongTensor(labels_of_gene)
    n_class = max(labels_of_gene).item() + 1
    features = torch.FloatTensor(gene_data)

    final_list = []
    for i in range(10):
        node_Idx = shuffle(np.arange(labels_of_gene.shape[0]), random_state=72)
        KF = KFold(n_splits=5)
        split_data = KF.split(node_Idx)
        acc_list = []
        for train_idx, test_idx in split_data:
            train_idx = torch.LongTensor(train_idx)
            test_idx = torch.LongTensor(test_idx)
            features = features.cpu()
            train_fts = features[train_idx]
            test_fts = features[test_idx]
            featured_train_idx = train_idx[(
                    train_fts.sum(1) != 0).nonzero().reshape([-1])]
            featured_test_idx = test_idx[(
                    test_fts.sum(1) != 0).nonzero().reshape([-1])]
            non_featured_test_idx = test_idx[(
                    test_fts.sum(1) == 0).nonzero().reshape([-1])]
            featured_train_lbls = labels_of_gene[featured_train_idx]
            featured_test_lbls = labels_of_gene[featured_test_idx]
            non_featured_test_lbls = labels_of_gene[non_featured_test_idx]
            featured_test_lbls_arr = featured_test_lbls.numpy()
            non_featured_test_lbls_arr = non_featured_test_lbls.numpy()
            model = GCN_eva(
                nfeat=features.shape[1], nhid=64, nclass=n_class, dropout=0.1, input_fts_sparse=False)
            if is_cuda:
                model.cuda()
                adj = adj.cuda()
                features = features.cuda()
                featured_train_lbls = featured_train_lbls.cuda()
                featured_test_lbls = featured_test_lbls.cuda()
                featured_train_idx = featured_train_idx.cuda()
                featured_test_idx = featured_test_idx.cuda()
            optimizer = optim.Adam(
                model.parameters(), lr=0.001, weight_decay=1e-5)
            best_acc = 0
            for epoch in range(1000):
                model.train()
                optimizer.zero_grad()
                output = model(features, adj)
                loss_train = F.nll_loss(
                    output[featured_train_idx], featured_train_lbls)
                loss_train.backward()
                optimizer.step()
                model.eval()
                val_loss = F.nll_loss(
                    output[featured_test_idx], featured_test_lbls)
                if is_cuda:
                    featured_preds = np.argmax(
                        output[featured_test_idx].data.cpu().numpy(), axis=1)
                else:
                    featured_preds = np.argmax(
                        output[featured_test_idx].data.numpy(), axis=1)
                random_preds = np.random.choice(
                    np.arange(n_class), len(non_featured_test_idx))
                preds = np.concatenate((featured_preds, random_preds))
                lbls = np.concatenate(
                    (featured_test_lbls_arr, non_featured_test_lbls_arr))
                acc = np.sum(preds == lbls) * 1.0 / len(preds)
                if acc > best_acc:
                    best_acc = acc

            acc_list.append(best_acc)
        avg_acc = np.mean(acc_list)
        final_list.append(avg_acc)
    print('GCN(A+X), avg accuracy: {}'.format(np.mean(final_list)))
    return np.mean(final_list)


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
