import torch.nn as nn
import dgl


class HeteroGraphConv(nn.Module):
    r"""A generic module for computing convolution on heterogeneous graphs.

    The heterograph convolution applies sub-modules on their associating
    relation graphs, which reads the features from source nodes and writes the
    updated ones to destination nodes.

    If the relation graph has no edge, the corresponding module will not be called.

    Parameters
    ----------
    mods : dict[str, nn.Module]
    """

    def __init__(self, mods: dict):
        super(HeteroGraphConv, self).__init__()
        self.mods = nn.ModuleDict(mods)

    def forward(self, g: dgl.DGLHeteroGraph, input_src: dict, input_dst: dict, node_transformation_weight: nn.ParameterDict, nodes_attention_weight: nn.ParameterDict):
        """
        call the forward function with each module.

        Parameters
        ----------
        g : DGLHeteroGraph
            The Heterogeneous Graph.
        input_src : dict[str, Tensor], Input source node features {'ntype': features, }.
        input_dst : dict[str, Tensor], Input destination node features {'ntype': features, }.
        node_transformation_weight: nn.ParameterDict, weights {'ntype', (inp_dim, hidden_dim)}
        nodes_attention_weight: nn.ParameterDict, weights {'ntype', (n_heads, 2 * hidden_dim)}

        Returns
        -------
        outputs, dict[str, Tensor]
            Output representations for every relation -> {(stype, etype, dtype): features}.
        """

        # key: (srctype, etype, dsttype), value: representations
        outputs = dict()

        for stype, etype, dtype in g.canonical_etypes:
            rel_graph = g[stype, etype, dtype]
            if rel_graph.number_of_edges() == 0:
                continue

            dst_representation = self.mods[etype](rel_graph,
                                                  (input_src[stype], input_dst[dtype]),
                                                  node_transformation_weight[dtype],
                                                  node_transformation_weight[stype],
                                                  nodes_attention_weight[stype])

            # dst_representation (dst_nodes, hid_dim)
            outputs[(stype, etype, dtype)] = dst_representation

        return outputs
