import torch
import torch.nn as nn
import warnings
import copy
import sys

from utils.utils import set_random_seed, convert_to_gpu, load_dataset
from utils.utils import get_n_params, get_ogb_evaluator
from model.HGConv import HGConv
from utils.Classifier import Classifier

args = {
    'dataset': 'OGB_MAG',
    'model_name': 'HGConv_lr0.001_dropout0.5_seed_0',
    'predict_category': 'paper',
    'seed': 0,
    'cuda': 0,
    'learning_rate': 0.001,
    'num_heads': 8,  # Number of attention heads
    'hidden_units': 32,
    'dropout': 0.5,
    'n_layers': 2,
    'residual': True
}
args['data_path'] = f'../dataset/{args["dataset"]}/{args["dataset"]}.pkl'
args['data_split_idx_path'] = f'../dataset/{args["dataset"]}/{args["dataset"]}_split_idx.pkl'
args['device'] = f'cuda:{args["cuda"]}' if torch.cuda.is_available() and args["cuda"] >= 0 else 'cpu'

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    set_random_seed(args['seed'])

    print(f'loading dataset {args["dataset"]}...')

    graph, labels, num_classes, train_idx, valid_idx, test_idx = load_dataset(data_path=args['data_path'],
                                                                              predict_category=args['predict_category'],
                                                                              data_split_idx_path=args[
                                                                                  'data_split_idx_path'])

    hgconv = HGConv(graph=graph,
                    input_dim_dict={ntype: graph.nodes[ntype].data['feat'].shape[1] for ntype in graph.ntypes},
                    hidden_dim=args['hidden_units'],
                    num_layers=args['n_layers'], n_heads=args['num_heads'], dropout=args['dropout'],
                    residual=args['residual'])

    classifier = Classifier(n_hid=args['hidden_units'] * args['num_heads'], n_out=num_classes)

    model = nn.Sequential(hgconv, classifier)

    model = convert_to_gpu(model, device=args['device'])
    print(model)

    print(f'Model #Params: {get_n_params(model)}')

    print(f'configuration is {args}')

    save_model_path = f"../save_model/{args['dataset']}/{args['model_name']}/{args['model_name']}.pkl"

    # load model parameter
    model.load_state_dict(torch.load(save_model_path, map_location='cpu'))

    print('performing model inference for test data...')
    # evaluate the best model
    model.eval()

    nodes_representation = model[0].inference(graph, copy.deepcopy(
        {ntype: graph.nodes[ntype].data['feat'] for ntype in graph.ntypes}), device=args['device'])

    test_y_predicts = model[1](convert_to_gpu(nodes_representation[args['predict_category']], device=args['device']))[
        test_idx]
    test_y_trues = convert_to_gpu(labels[test_idx], device=args['device'])
    test_accuracy = get_ogb_evaluator(predicts=test_y_predicts.argmax(dim=1),
                                                 labels=test_y_trues)
    print(f'final test accuracy {test_accuracy:.4f}')

    sys.exit()
