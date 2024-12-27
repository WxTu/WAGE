from tqdm import tqdm
from utils.registry import TRAIN_REGISTRY
from utils.misc import *
from utils.loss_funcs import *
from utils.evaluation import *
from torch_geometric.utils import negative_sampling, dense_to_sparse
from dgl import RemoveSelfLoop


@TRAIN_REGISTRY.register()
def Train_wage(args, model, dgl_graph, pyg_graph, train_fts_idx, test_fts_idx, vali_test_idx,
               mode='train'):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    remove = RemoveSelfLoop()
    graph_tmp = remove(dgl_graph)
    dense_adj = graph_tmp.adjacency_matrix().to_dense()
    adj_scope = compute_high_order_adj(dense_adj, args.max_scope)

    adj_all = 0
    for adj_i in adj_scope:
        adj_all += adj_i

    sub_graph = dgl.remove_edges(dgl_graph, vali_test_idx)
    sub_graph_tmp = remove(sub_graph)
    sub_dense_adj = sub_graph_tmp.adjacency_matrix().to_dense()
    multi_order_adj = compute_high_order_adj(sub_dense_adj, args.max_order)

    graph_index = []
    subgraph_index = []
    for i in range(0, args.max_order):
        graph_index.append(dense_to_sparse(adj_scope[i])[0].to(device))
        subgraph_index.append(dense_to_sparse(multi_order_adj[i])[0].to(device))

    true_features = copy.deepcopy(dgl_graph.ndata['feat'])
    pyg_graph.x[vali_test_idx] = 0.0
    train_fts = true_features[train_fts_idx]
    pos_weight = torch.sum(true_features[train_fts_idx] == 0.0).item(
    ) / (torch.sum(true_features[train_fts_idx] != 0.0).item())

    pos_edges = torch.stack(dgl_graph.edges(), dim=0)
    neg_edges = negative_sampling(
        pos_edges,
        num_nodes=dgl_graph.num_nodes(),
        num_neg_samples=pos_edges.size(1),
    ).view_as(pos_edges)

    subpos_edges = torch.stack(sub_graph.edges(), dim=0)
    subneg_edges = negative_sampling(
        subpos_edges,
        num_nodes=dgl_graph.num_nodes(),
        num_neg_samples=subpos_edges.size(1),
    ).view_as(subpos_edges)

    adj_all = adj_all.to(device)
    pyg_graph = pyg_graph.to(device)
    true_features = true_features.to(device)
    train_fts = train_fts.to(device)
    pos_edges = pos_edges.to(device)
    neg_edges = neg_edges.to(device)
    subpos_edges = subpos_edges.to(device)
    subneg_edges = subneg_edges.to(device)
    pos_weight_tensor = torch.FloatTensor([pos_weight]).to(device)
    neg_weight_tensor = torch.FloatTensor([1.0]).to(device)

    if args.dataset_type == 'one-hot':
        loss_function = loss_function_discrete
    else:
        loss_function = loss_function_continuous

    if mode != 'train':
        model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'save', 'model_param', args.method,
                                                      'final_{}.pkl'.format(args.dataset))))
        model.eval()
        with torch.no_grad():
            output, embedding = model.embed(pyg_graph, graph_index, adj_all)
        return output, embedding

    for _ in tqdm(range(args.epoch)):
        model.train()

        optimizer.zero_grad()
        X_hat, Zs = model(pyg_graph, subgraph_index, adj_all)

        pos = Zs[pos_edges[0]] * Zs[pos_edges[1]]
        pos = pos.sum(dim=-1)
        neg = Zs[neg_edges[0]] * Zs[neg_edges[1]]
        neg = neg.sum(dim=-1)

        subpos = Zs[subpos_edges[0]] * Zs[subpos_edges[1]]
        subpos = subpos.sum(dim=-1)
        subneg = Zs[subneg_edges[0]] * Zs[subneg_edges[1]]
        subneg = subneg.sum(dim=-1)

        La = args.lambda_xr * loss_function(X_hat[train_fts_idx], train_fts, pos_weight_tensor, neg_weight_tensor)

        Ls = args.lambda_ar * ce_loss(pos, neg)
        sub_Ls = args.lambda_sub_ar * ce_loss(subpos, subneg)

        L = La + sub_Ls + Ls
        L.backward(retain_graph=True)
        optimizer.step()

        with torch.no_grad():
            model.eval()
            output, _ = model(pyg_graph, graph_index, adj_all)

    torch.save(model.state_dict(), os.path.join(os.getcwd(), 'save', 'model_param', args.method,
                                                'final_{}.pkl'.format(args.dataset)))

    model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'save', 'model_param', args.method,
                                                  'final_{}.pkl'.format(args.dataset))))
    with torch.no_grad():
        model.eval()
        output, _ = model(pyg_graph, graph_index, adj_all)

    gene_test_fts = output[test_fts_idx].data.cpu().numpy()
    gt_fts = true_features[test_fts_idx].cpu().numpy()
    print('Method: {}, Dataset: {}'.format(args.method, args.dataset))
    for topK in args.topK_list:
        avg_recall, avg_ndcg = RECALL_NDCG(gene_test_fts, gt_fts, topN=topK)
        print('topK: {}, Recall: {}, NDCG: {}'.format(topK, avg_recall, avg_ndcg))