from utils.misc import *
from datasets.pyg_data_utils import load_pyg_dataset
from models import build_model
from trains import pre_train
from utils.evaluation import *
import argparse
import GPUtil
import time


def build_args():
    parser = argparse.ArgumentParser(
        description="Attribute-missing Graph Learning")
    parser.add_argument("--dataset", type=str, default="cs")
    parser.add_argument("--method", type=str, default="wage")
    parser.add_argument("--frame", type=str, default="pyg")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--max_order", type=int, default=3)
    parser.add_argument("--max_scope", type=int, default=5)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--resume", type=int, help='1 or 0', default=1)
    parser.add_argument("--split_seed", type=float, default=72)
    parser.add_argument("--train_fts_ratio", type=float, default=0.4)
    parser.add_argument("--drop_ratio", type=float, default=0.6)
    parser.add_argument('--topK_list', type=list, default=[10, 20, 50])
    parser.add_argument("--encoder_name", type=str, help='GCN/GAT', default="GCN")
    parser.add_argument("--dataset_type", type=str, default="one-hot")
    parser.add_argument("--cuda", type=bool, default=True)
    args = parser.parse_args()
    return args


def main(args):
    global profiling_feature
    if args.device == -1:
        while True:
            try:
                devices = GPUtil.getAvailable(order='memory')

                if len(devices) == 0:
                    raise RuntimeError("No available GPUs, wait for 10 minutes")
                device_id = devices[0]
                print('Use GPU:', device_id)
                break
            except Exception as e:
                print("Error:", e)
                time.sleep(600)
    else:
        device_id = args.device

    torch.cuda.set_device(device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pyg_graph, num_nodes, num_features, num_classes = load_pyg_dataset(
        args.dataset)
    dgl_graph = pyg2dgl(pyg_graph)

    if args.dataset in ['pubmed', 'cs']:
        args.dataset_type = 'continuous'

    args.num_nodes = num_nodes
    args.num_features = num_features
    train_id, _, test_id, vali_test_id = data_split(args, num_nodes)

    model = build_model(args)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model total parameters: {total_params}")
    model = model.to(device)

    if args.resume:
        train_func = pre_train(args)
        print('------------------------------start pre_training-------------------------------')
        train_func(args, model, dgl_graph, pyg_graph, train_id, test_id, vali_test_id)
        model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'save', 'model_param', args.method,
                                                      'final_{}.pkl'.format(args.dataset))))
        gene_data, z = train_func(args, model, dgl_graph, pyg_graph, train_id, test_id, vali_test_id,
                                  mode='inference')
        feat = torch.sigmoid(gene_data[test_id]).cpu().numpy()
    else:
        print('---------------------------load model and testing------------------------------')
        train_func = pre_train(args)
        model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'save', 'model_param', args.method,
                                                      'final_{}.pkl'.format(args.dataset))))
        gene_data, z = train_func(args, model, dgl_graph, pyg_graph, train_id, test_id, vali_test_id,
                                  mode='inference')
        profiling_feature = gene_data[test_id].cpu().numpy()
        feat = torch.sigmoid(gene_data[test_id]).cpu().numpy()
        test_Profiling(profiling_feature, dgl_graph.ndata['feat'][test_id].detach().cpu().numpy())


if __name__ == "__main__":
    args = build_args()
    if args.method in ['wage']:
        args = load_best_configs(args, f"configs/{args.method}/configs.yml")
    print(args)
    set_random_seed(args.seed)
    main(args)
