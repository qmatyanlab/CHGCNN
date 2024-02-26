
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import aggr



import time

class CHGInterConv(MessagePassing):
    def __init__(self, node_fea_dim=92, hedge_fea_dim=35, batch_norm = True):
        super().__init__()
        self.batch_norm = batch_norm
        self.node_fea_dim = node_fea_dim
        self.hedge_fea_dim = hedge_fea_dim

        self.lin_full_presplit = Linear(2*node_fea_dim+hedge_fea_dim, 2*node_fea_dim)
        self.lin_filter = Linear(2*node_fea_dim+hedge_fea_dim, node_fea_dim)
        self.lin_core = Linear(2*node_fea_dim+hedge_fea_dim, node_fea_dim)

        self.softplus_hedge = torch.nn.Softplus()
        self.sigmoid_filter = torch.nn.Sigmoid()
        self.softplus_core = torch.nn.Softplus()
        self.softplus_out = torch.nn.Softplus()


        self.node_aggr = aggr.MeanAggregation()

        if batch_norm == True:
            self.bn_1 = BatchNorm1d(2*node_fea_dim)
            self.bn_2 = BatchNorm1d(node_fea_dim)

    def forward(self, x, inter_relations_index, connect_feats, num_nodes):
        '''
        x:              torch tensor (of type float) of node attributes

                        [[node1_feat],[node2_feat],...]
                        dim([num_nodes,node_fea_dim])

        inter_relations_index:    torch tensor (of type long) of
                        hyperedge indices (as in HypergraphConv)

                        [[node_indxs,...],[hyperedge_indxs,...],[node_indxs,...]]
                        dim([3,num nodes in all hedges])

        hedge attr:     torch tensor (of type float) of
                        hyperedge attributes (with first index algining with 
                        hedges overall hyperedge_indx in hedge_index)
  
                        [[hedge1_feat], [hedge2_feat],...]
                        dim([num_hedges,hyperedge_feat_dim])

        '''
        origin_xs = x[inter_relations_index[0]]
        connect_xs = connect_feats[inter_relations_index[1]]
        remote_xs = x[inter_relations_index[0]]

        z = torch.cat([origin_xs, connect_xs, remote_xs], dim = 1)
        

        z = self.lin_full_presplit(z)
        if self.batch_norm == True:
            z = self.bn_1(z)
        z_f, z_c = z.chunk(2, dim = -1)
        out = self.sigmoid_filter(z_f)*self.softplus_core(z_c)# Apply CGConv like structure
        out = self.node_aggr(out, inter_relations_index[0], dim_size = num_nodes, dim = 0) #aggregate according to node
 
        if self.batch_norm == True:
            out = self.bn_2(out)

        out = self.softplus_out(out + x)

        return out