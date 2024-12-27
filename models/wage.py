from utils.registry import MODEL_REGISTRY
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


@MODEL_REGISTRY.register()
class wage(nn.Module):
    def __init__(self, opt):
        super().__init__()
        hidden1 = opt.hidden1
        hidden2 = opt.hidden2
        out_features = opt.num_features
        self.neighbor_num = opt.neighbor_num
        self.dropout = 0.5
        self.sig = nn.Sigmoid()

        self.shared_e1 = GCNConv(out_features, hidden1, cached=True, bias=False)
        self.shared_e2 = GCNConv(hidden1, hidden2, cached=True, bias=False)

        self.shared_d1 = GCNConv(hidden2, hidden1, cached=True, bias=False)
        self.shared_d2 = GCNConv(hidden1, out_features, cached=True, bias=False)

        self.fusion = Fusion(opt)

        self.alpha = nn.Parameter(nn.init.constant_(torch.zeros(1), opt.fusion_weight), requires_grad=True)
        self.beta = nn.Parameter(nn.init.constant_(torch.zeros(1), opt.fusion_weight), requires_grad=True)

    def DNA(self, x, edge_index, I):
        Z = F.elu(self.shared_e1(x, edge_index))
        Z = F.dropout(Z, self.dropout, training=self.training)
        Z = F.elu(self.shared_e2(Z, edge_index))

        Z_norm = F.normalize(Z)
        S = torch.mm(Z_norm, Z_norm.t())

        topk_g, _ = torch.topk(S, self.neighbor_num, dim=1, largest=True)
        S_g = self.rebuilt_adj(S, topk_g)
        topk_l, _ = torch.topk(torch.where(I == 0, -torch.ones_like(S), S), self.neighbor_num, dim=1, largest=True)
        S_l = self.rebuilt_adj(torch.where(I == 0, -torch.ones_like(S), S), topk_l)

        Za = torch.mm(S_g, Z) + (1 - self.alpha) * torch.mm(S_l, Z)
        return Za

    def HSE(self, x, edge_index):
        Z = F.relu(self.shared_e1(x, edge_index))
        Z = F.dropout(Z, self.dropout, training=self.training)
        Z = F.elu(self.shared_e2(Z, edge_index))
        return Z

    def shared_decoder(self, x, edge_index):
        Z = F.relu(self.shared_d1(x, edge_index))
        Z = F.dropout(Z, self.dropout, training=self.training)
        X_hat = F.elu(self.shared_d2(Z, edge_index))
        return X_hat

    def rebuilt_adj(self, S, topk):
        topk_min = torch.min(topk, dim=-1).values.unsqueeze(-1).repeat(1, S.shape[-1])
        H = F.softmax(torch.where(torch.ge(S, topk_min), S, torch.zeros_like(S) - 10000), dim=1)
        return H

    def forward(self, graph, A_set, I):
        X, edge_index = graph.x, graph.edge_index
        X = F.dropout(X, self.dropout, training=self.training)
        Za = self.DNA(X, edge_index, I)
        Z_box = []
        for i in range(len(A_set)):
            Z_box.append((self.HSE(X, A_set[i])))
        Zs, _ = self.fusion(Z_box)
        Zf = self.beta * Zs + (1 - self.beta) * Za
        X_hat = self.shared_decoder(Zf, edge_index)
        return X_hat, Zs

    def embed(self, graph, A_set, I):
        X, edge_index = graph.x, graph.edge_index
        X = F.dropout(X, self.dropout, training=self.training)
        Za = self.DNA(X, edge_index, I)
        Z_box = []
        for i in range(len(A_set)):
            Z_box.append((self.HSE(X, A_set[i])))
        Zs, _ = self.fusion(Z_box)
        Zf = self.beta * Zs + (1 - self.beta) * Za
        X_hat = self.shared_decoder(Zf, edge_index)
        return X_hat, Zs


class Fusion(nn.Module):
    def __init__(self, args):
        super(Fusion, self).__init__()
        self.args = args
        self.A = nn.ModuleList([nn.Linear(args.hidden2, 1) for _ in range(self.args.max_order)])
        self.weight_init()

    def weight_init(self):
        for i in range(self.args.max_order):
            nn.init.xavier_normal_(self.A[i].weight)
            self.A[i].bias.data.fill_(0.0)

    def forward(self, feat_pos):
        feat_pos, feat_pos_attn = self.attn_feature(feat_pos)
        return feat_pos, feat_pos_attn

    def attn_feature(self, features):
        features_attn = []
        for i in range(self.args.max_order):
            features_attn.append((self.A[i](features[i])))
        features_attn = F.softmax(torch.cat(features_attn, 1), -1)
        features = torch.cat(features, 0)
        features_attn_reshaped = features_attn.transpose(1, 0).contiguous().view(-1, 1)
        features = features * features_attn_reshaped.expand_as(features)
        features = features.view(self.args.max_order, self.args.num_nodes, self.args.hidden2).sum(0)
        return features, features_attn
