
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import aggr


class CHGConv(MessagePassing):
    def __init__(self, node_fea_dim=92, hedge_fea_dim=35, batch_norm = True):
        super().__init__()
        self.batch_norm = batch_norm
        self.node_fea_dim = node_fea_dim
        self.hedge_fea_dim = hedge_fea_dim

        self.lin_f1 = Linear(node_fea_dim+hedge_fea_dim, hedge_fea_dim+node_fea_dim)
        self.lin_c1 = Linear(node_fea_dim+hedge_fea_dim, hedge_fea_dim)
        self.lin_f2 = Linear(2*node_fea_dim+hedge_fea_dim, 2*node_fea_dim)

        self.softplus_hedge = torch.nn.Softplus()
        self.sigmoid_filter = torch.nn.Sigmoid()
        self.softplus_core = torch.nn.Softplus()
        self.softplus_out = torch.nn.Softplus()


        self.hedge_aggr = aggr.SoftmaxAggregation(learn = True)
        self.node_aggr = aggr.SoftmaxAggregation(learn = True)

        if batch_norm == True:
            self.bn_f = BatchNorm1d(node_fea_dim)
            self.bn_c = BatchNorm1d(node_fea_dim)

            self.bn_o = BatchNorm1d(node_fea_dim)

    def forward(self, hyperedge_attrs_tuple, hyperedge_index):
        '''
        hyperedge_attrs_tuple:    tuple of torch tensor (of type float) of source and destination hyperedge attributes

                        ([hedge1_feat],...],[[node1_feat],[node2_feat],...)
                        (dim(hedge_feat_dim,num_hedges),dim(num_nodes, node_feat_dim))

        hedge_index:    torch tensor (of type long) of
                        hyperedge indices (as in HypergraphConv)

                        [[node_indxs,...],[hyperedge_indxs,...]]
                        dim([2,num nodes in all hedges])


        '''

        '''
        The goal is to generalize the CGConv gated convolution structure to hyperedges. The 
        primary problem with such a generalization is the variable number of nodes contained 
        in each hyperedge (hedge). I propose we simply aggregate the nodes contained within 
        each hedge to complete the message, and then concatenate that with the hyperedge feature 
        to form the message.

        Below, the node attributes are first placed in order with their hyperedge_indices
        and then aggregated according to their hyperedges to form a component of the message corresponding to 
        each hyperedge
        '''
        hedge_attr, x = hyperedge_attrs_tuple
        num_nodes = x.shape[0]
        num_hedges = hedge_attr.shape[0]
        hedge_index_xs = x[hyperedge_index[1].int()]
        hedge_index_xs = self.hedge_aggr(hedge_index_xs, hyperedge_index[0], dim_size = num_hedges)

        '''
        To finish forming the message, I concatenate these aggregated neighborhoods with their 
        corresponding hedge features.
        '''

        message_holder = torch.cat([hedge_index_xs, hedge_attr], dim = 1)
        '''
        We then can aggregate the messages and add to node features after some activation 
        functions and linear layers.
        '''
        hyperedge_attrs = self.lin_c1(message_holder)
        hyperedge_attrs = self.softplus_hedge(hedge_attr + hyperedge_attrs)
        message_holder = self.lin_f1(message_holder)
        x_i = x[hyperedge_index[1]]  # Target node features
        x_j = message_holder[hyperedge_index[0]]  # Source node features
        z = torch.cat([x_i,x_j], dim=-1)  # Form reverse messages (for origin node messages)
        z = self.lin_f2(z)
        z_f, z_c = z.chunk(2, dim = -1)
        if self.batch_norm == True:
            z_f = self.bn_f(z_f)
            z_c = self.bn_c(z_c)
        out = self.sigmoid_filter(z_f)*self.softplus_core(z_c) # Apply CGConv like structure
        out = self.node_aggr(out, hyperedge_index[1], dim_size = num_nodes) #aggregate according to node
 
        if self.batch_norm == True:
            out = self.bn_o(out)

        out = self.softplus_out(out + x)

        return out
