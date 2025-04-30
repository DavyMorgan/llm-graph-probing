import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class GCNRegressorLinear(nn.Module):
    def __init__(self, num_nodes, hidden_channels, out_channels, num_layers=2, dropout=0.0):
        super(GCNRegressorLinear, self).__init__()
        self.embedding = nn.Embedding(num_nodes, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, add_self_loops=False, normalize=False))
        self.fc1 = nn.Linear(2*hidden_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, 1)
        self.dropout = dropout

    def forward_graph_embedding(self, x, edge_index, edge_weight, batch):
        x = self.embedding(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = F.dropout(x, p=self.dropout, training=self.training)
        mean_x = global_mean_pool(x, batch)
        max_x = global_max_pool(x, batch)
        x = torch.cat([mean_x, max_x], dim=1)
        return x

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.forward_graph_embedding(x, edge_index, edge_weight, batch)
        x = self.fc2(self.fc1(x))
        return x


class GCNRegressor(nn.Module):
    def __init__(self, num_nodes, hidden_channels, out_channels, num_layers=2, dropout=0.0):
        super(GCNRegressor, self).__init__()
        self.embedding = nn.Embedding(num_nodes, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, add_self_loops=False, normalize=False))
        self.fc1 = nn.Linear(2*hidden_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, 1)
        self.dropout = dropout

    def forward_graph_embedding(self, x, edge_index, edge_weight, batch):
        x = self.embedding(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        mean_x = global_mean_pool(x, batch)
        max_x = global_max_pool(x, batch)
        x = torch.cat([mean_x, max_x], dim=1)
        return x

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.forward_graph_embedding(x, edge_index, edge_weight, batch)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
