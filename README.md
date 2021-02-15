# Hybrid Micro/Macro Level Convolution for Heterogeneous Graph Learning

This experiment is based on stanford OGB (1.2.4) benchmark. 
The description of "Hybrid Micro/Macro Level Convolution for Heterogeneous Graph Learning" is [avaiable here](https://arxiv.org/abs/2012.14722). 
The steps are:

  1. run ```python preprocess_ogbn_mag.py``` to preprocess the original ogbn_mag dataset. 
  As the OGB-MAG dataset only has input features for paper nodes, ccfor all the other types of nodes (author, affiliation, topic), we use the metapath2vec model to generate their structural features. 

  2. run ```python train_ogbn_mag.py``` to train the model.

  3. run ```python eval_ogbn_mag.py``` to evaluate the model.

## Environments:
- [PyTorch 1.7.0](https://pytorch.org/)
- [DGL 0.5.2](https://www.dgl.ai/)
- [PyTorch Geometric 1.6.1](https://pytorch-geometric.readthedocs.io/en/latest/)
- [tqdm](https://github.com/tqdm/tqdm)
- [numpy](https://github.com/numpy/numpy)

## Selected hyperparameters:

```
  num_heads              INT      Number of attention heads                          8
  hidden_units           INT      Dimension of hidden units for each head            32
  n_layers               INT      Number of GNN layers                               2
  learning_rate          FLOAT    Learning rate                                      0.001
  dropout                FLOAT    Dropout rate                                       0.5
  residual               BOOL     Whether to use the residual connection             True
```

Hyperparameters could be found in the ```args``` variable in ```train_ogbn_mag.py``` file and you can adjust them when training the model.
When evaluating the model, please make sure the ```args``` in ```eval_ogbn_mag.py``` keep the same to those in the training process.

## Reference performance for the OGB-MAG dataset:

| Model        | Test Accuracy   | Valid Accuracy  | # Parameter     | Hardware         |
| ---------    | --------------- | --------------  | --------------  |--------------    |
| HGConv  | 0.5045 ± 0.0017   | 0.5300 ± 0.0018  |    2,850,405      | NVIDIA TITAN Xp (12GB) |

## Citation
Please consider citing our paper when using the code.

```bibtex
@article{yu2020hybrid,
  title={Hybrid Micro/Macro Level Convolution for Heterogeneous Graph Learning},
  author={Yu, Le and Sun, Leilei and Du, Bowen and Liu, Chuanren and Lv, Weifeng and Xiong, Hui},
  journal={arXiv preprint arXiv:2012.14722},
  year={2020}
}
```
