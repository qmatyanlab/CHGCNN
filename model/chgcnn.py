import torch
import numpy as np
from torch_scatter import scatter
from torch_geometric.nn.conv import HeteroConv, HypergraphConv
import torch_geometric.nn as nn
import torch

from .convolutions.inter_conv import CHGInterConv
from .convolutions.agg_conv import CHGConv



class CrystalHypergraphConv(torch.nn.Module):
    def __init__(self, classification, h_dim = 64, hout_dim = 128, hidden_hedge_dim = 128, n_layers = 1):
        super().__init__()

        self.classification = classification

        bond_hedge_dim = 40
        motif_hedge_dim = 94 
        triplet_hedge_dim = 40
        self.embed = torch.nn.Embedding(101, h_dim)
        self.bembed = nn.Linear(bond_hedge_dim, hidden_hedge_dim)
        self.membed = nn.Linear(motif_hedge_dim, hidden_hedge_dim)
        self.convs_btb = torch.nn.ModuleList() 
        self.convs = torch.nn.ModuleList() 
        for _ in range(n_layers):
            conv = HeteroConv({
#                ('bond', 'contains', 'atom'): CHGConv(node_fea_dim = h_dim, hedge_fea_dim = bond_hedge_dim),
#                ('triplet', 'contains', 'atom'): CHGConv(node_fea_dim = h_dim, hedge_fea_dim = triplet_hedge_dim),
                ('motif', 'contains', 'atom'): CHGConv(node_fea_dim = h_dim, hedge_fea_dim = motif_hedge_dim),
#                ('atom', 'in', 'bond'): CHGConv(node_fea_dim = bond_hedge_dim, hedge_fea_dim = h_dim),
#                ('atom', 'in', 'motif'): CHGConv(node_fea_dim = motif_hedge_dim, hedge_fea_dim = h_dim),
#                ('bond', 'in', 'motif'): CHGConv(node_fea_dim = hedge_dim, hedge_fea_dim = hedge_dim),
#                ('motif', 'contains', 'bond'): CHGConv(node_fea_dim = hedge_dim, hedge_fea_dim = hedge_dim),
            })
            self.convs.append(conv)
        self.l1 = nn.Linear(h_dim, hout_dim)
        self.activation = torch.nn.Softplus()
        if self.classification:
            self.out = nn.Linear(hout_dim, 2)
            self.sigmoid = torch.nn.Sigmoid()
            self.dropout = torch.nn.Dropout()
        else:
            self.out = nn.Linear(hout_dim,1)
 
    def forward(self, data):
        hyperedge_attrs_dict = data.hyperedge_attrs_dict
        hyperedge_index_dict = data.hyperedge_index_dict
        batch = data['atom'].batch
        hyperedge_attrs_dict['atom'] = self.embed(hyperedge_attrs_dict['atom'].long())
        for conv in self.convs:
            hyperedge_attrs_dict_update = conv(hyperedge_attrs_dict, hyperedge_index_dict)
            hyperedge_attrs_dict['atom'] = hyperedge_attrs_dict_update['atom'].relu()
        x = scatter(hyperedge_attrs_dict['atom'], batch, dim=0, reduce='mean')
        x = self.l1(x)
        if self.classification:
            x = self.dropout(x)
        x = self.activation(x)
        output = self.out(x)
        if self.classification:
            output = self.sigmoid(output)
        return output


       
