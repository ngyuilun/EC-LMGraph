


import torch
import torch_geometric
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, LEConv, global_mean_pool, ClusterGCNConv




class LE_ClusterGCN_l1_fc2_lm(torch.nn.Module):

    def __init__(self, args):
        super(LE_ClusterGCN_l1_fc2_lm, self).__init__()

        dim=args.dim


        self.input = None
        self.final_conv_acts = None
        self.final_conv_grads = None

        # layers
        self.le_conv1 = LEConv(args.num_features, dim)
        self.conv1 = ClusterGCNConv(dim, dim)
        self.fc1 = torch.nn.Linear(dim, dim)
        self.fc2 = torch.nn.Linear(dim, args.num_classes)
        
        

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, x, edge_index, batch, edge_weight=None):
        h0 = x
        # h0.requires_grad = True
        self.input = h0
        h1 = self.le_conv1(h0, edge_index).relu()
        with torch.enable_grad():
            self.final_conv_acts = self.conv1(h1, edge_index)
        self.final_conv_acts.register_hook(self.activations_hook)
        h3 = self.final_conv_acts.relu()
        h4 = global_mean_pool(h3, batch)
        h4 = self.fc1(h4).relu()
        out = self.fc2(h4)
        return out
    











