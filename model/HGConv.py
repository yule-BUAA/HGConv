import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from tqdm import tqdm

from model.MicroConvolution import MicroConv
from model.HeteroConv import HeteroGraphConv
from model.MacroConvolution import MacroConv


class HGConvLayer(nn.Module):
    def __init__(self, graph: dgl.DGLHeteroGraph, input_dim: int, hidden_dim: int, n_heads: int = 4,
                 dropout: float = 0.2, residual: bool = True):
        """

        :param graph: a heterogeneous graph
        :param input_dim: int, input dimension
        :param hidden_dim: int, hidden dimension
        :param n_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param residual: boolean, residual connections or not
        """
        super(HGConvLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.residual = residual

        # same type neighbors aggregation trainable parameters
        self.node_transformation_weight = nn.ParameterDict({
            ntype: nn.Parameter(torch.randn(input_dim, n_heads * hidden_dim))
            for ntype in graph.ntypes
        })
        self.nodes_attention_weight = nn.ParameterDict({
            ntype: nn.Parameter(torch.randn(n_heads, 2 * hidden_dim))
            for ntype in graph.ntypes
        })

        # different types aggregation trainable parameters
        self.edge_type_transformation_weight = nn.ParameterDict({
            etype: nn.Parameter(torch.randn(n_heads * hidden_dim, n_heads * hidden_dim))
            for _, etype, _ in graph.canonical_etypes
        })

        self.central_node_transformation_weight = nn.ParameterDict({
            ntype: nn.Parameter(torch.randn(input_dim, n_heads * hidden_dim))
            for ntype in graph.ntypes
        })

        self.edge_types_attention_weight = nn.Parameter(torch.randn(n_heads, 2 * hidden_dim))

        # hetero conv modules
        self.micro_conv = HeteroGraphConv({
            etype: MicroConv(in_feats=(input_dim, input_dim), out_feats=hidden_dim,
                           num_heads=n_heads, dropout=dropout, negative_slope=0.2)
            for srctype, etype, dsttype in graph.canonical_etypes
        })

        # different types aggregation module
        self.macro_conv = MacroConv(in_feats=hidden_dim * n_heads, out_feats=hidden_dim,
                                                             num_heads=n_heads,
                                                             dropout=dropout, negative_slope=0.2)

        if self.residual:
            # residual connection
            self.res_fc = nn.ModuleDict()
            self.residual_weight = nn.ParameterDict()
            for ntype in graph.ntypes:
                self.res_fc[ntype] = nn.Linear(input_dim, n_heads * hidden_dim, bias=True)
                self.residual_weight[ntype] = nn.Parameter(torch.randn(1))

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        for weight in self.node_transformation_weight:
            nn.init.xavier_normal_(self.node_transformation_weight[weight], gain=gain)
        for weight in self.nodes_attention_weight:
            nn.init.xavier_normal_(self.nodes_attention_weight[weight], gain=gain)
        for weight in self.edge_type_transformation_weight:
            nn.init.xavier_normal_(self.edge_type_transformation_weight[weight], gain=gain)
        for weight in self.central_node_transformation_weight:
            nn.init.xavier_normal_(self.central_node_transformation_weight[weight], gain=gain)

        nn.init.xavier_normal_(self.edge_types_attention_weight, gain=gain)
        if self.residual:
            for ntype in self.res_fc:
                nn.init.xavier_normal_(self.res_fc[ntype].weight, gain=gain)

    def forward(self, graph: dgl.DGLHeteroGraph, node_features: dict):
        """

        :param graph: dgl.DGLHeteroGraph
        :param node_features: dict, {"type": features}
        :return: output_features: dict, {"type": features}
        """
        # dictionary of input source features and destination features
        input_src = node_features
        if graph.is_block:
            input_dst = {}
            for ntype in node_features:
                input_dst[ntype] = node_features[ntype][:graph.number_of_dst_nodes(ntype)]
        else:
            input_dst = node_features
        # same type neighbors aggregation
        # relation_features, dict, {(stype, etype, dtype): features}
        relation_features = self.micro_conv(graph, input_src, input_dst, self.node_transformation_weight,
                                             self.nodes_attention_weight)
        # different types aggregation
        output_features = self.macro_conv(graph, input_dst, relation_features,
                                                 self.edge_type_transformation_weight,
                                                 self.central_node_transformation_weight,
                                                 self.edge_types_attention_weight)

        if self.residual:
            for ntype in output_features:
                alpha = F.sigmoid(self.residual_weight[ntype])
                output_features[ntype] = output_features[ntype] * alpha + self.res_fc[ntype](input_dst[ntype]) * (1 - alpha)

        return output_features


class HGConv(nn.Module):
    def __init__(self, graph: dgl.DGLHeteroGraph, input_dim_dict: dict, hidden_dim: int, num_layers: int, n_heads: int = 4,
                 dropout: float = 0.2, residual: bool = True):
        """

        :param graph: a heterogeneous graph
        :param input_dim_dict: input dim dictionary
        :param hidden_dim: int, hidden dimension
        :param num_layers: int, number of stacked layers
        :param n_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param residual: boolean, residual connections or not
        """
        super(HGConv, self).__init__()

        self.input_dim_dict = input_dim_dict
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        # align the dimension of different types of nodes
        self.projection_layer = nn.ModuleDict({
            ntype: nn.Linear(input_dim_dict[ntype], hidden_dim * n_heads) for ntype in input_dim_dict
        })

        self.layers = nn.ModuleList()

        # hidden layers
        for _ in range(self.num_layers):
            self.layers.append(HGConvLayer(graph, hidden_dim * n_heads, hidden_dim, n_heads, dropout, residual))

    def forward(self, blocks: list, node_features: dict):
        """

        :param blocks: list of sampled dgl.DGLHeteroGraph
        :param node_features: node features, dict, {"type": features}
        :return:
        """
        # feature projection
        for ntype in node_features:
            node_features[ntype] = self.projection_layer[ntype](node_features[ntype])

        # graph convolution
        for block, layer in zip(blocks, self.layers):
            node_features = layer(block, node_features)
        return node_features

    def inference(self, graph: dgl.DGLHeteroGraph, node_features: dict, device: str):
        """
        mini-batch inference of final representation over all node types. Outer loop: Interate the layers, Inner loop: Interate the batches

        :param graph: The whole graph
        :param node_features: features of all the nodes in the whole graph, dict, {"type": features}
        :param device: device str
        """
        with torch.no_grad():
            # interate over each layer
            for index, layer in enumerate(self.layers):
                # Tensor, features of all types of nodes, store on cpu
                y = {
                    ntype: torch.zeros(
                        graph.number_of_nodes(ntype), self.hidden_dim * self.n_heads) for ntype in graph.ntypes}
                # full sample for each type of nodes
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                dataloader = dgl.dataloading.NodeDataLoader(
                    graph,
                    {ntype: torch.arange(graph.number_of_nodes(ntype)) for ntype in graph.ntypes},
                    sampler,
                    batch_size=1280,
                    shuffle=True,
                    drop_last=False,
                    num_workers=4)

                tqdm_dataloader = tqdm(dataloader, ncols=120)
                for batch, (input_nodes, output_nodes, blocks) in enumerate(tqdm_dataloader):
                    block = blocks[0].to(device)

                    input_features = {ntype: node_features[ntype][input_nodes[ntype]].to(device) for ntype in input_nodes.keys()}

                    if index == 0:
                        # feature projection for the first layer in the full batch inference
                        for ntype in input_features:
                            input_features[ntype] = self.projection_layer[ntype](input_features[ntype])

                    h = layer(block, input_features)

                    for k in h.keys():
                        y[k][output_nodes[k]] = h[k].cpu()

                    tqdm_dataloader.set_description(f'inference for the {batch}-th batch in model {index}-th layer')

                # update the features of all the nodes (after the graph convolution) in the whole graph
                node_features = y

        return y
