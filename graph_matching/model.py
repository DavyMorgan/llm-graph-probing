import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class GCNEncoder(nn.Module):
    def __init__(self, num_nodes, hidden_channels, out_channels, num_layers=1, dropout=0.0):
        super(GCNEncoder, self).__init__()
        self.embedding = nn.Embedding(num_nodes, hidden_channels)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, add_self_loops=False, normalize=False))
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.embedding(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        mean_x = global_mean_pool(x, batch)
        max_x = global_max_pool(x, batch)
        x = torch.cat([mean_x, max_x], dim=1)
        return x


class GraphMatchingModel(nn.Module):
    def __init__(
        self,
        num_nodes_llm_1,
        num_nodes_llm_2,
        hidden_channels,
        out_channels,
        num_layers=1,
        dropout=0.0,
        temperature=1.0
    ):
        super(GraphMatchingModel, self).__init__()
        self.encoder_llm_1 = GCNEncoder(num_nodes_llm_1, hidden_channels, out_channels, num_layers, dropout)
        self.encoder_llm_2 = GCNEncoder(num_nodes_llm_2, hidden_channels, out_channels, num_layers, dropout)
        self.temperature = temperature

    def forward(self, data):
        emb_llm, emb_human = self.forward_emb(data)
        sim_matrix = torch.matmul(emb_llm, emb_human.t()) / self.temperature
        return emb_llm, emb_human, sim_matrix

    def forward_emb(self, data):
        emb_llm_1 = self.encoder_llm_1(data.x_llm_1, data.edge_index_llm_1, data.edge_attr_llm_1, data.x_llm_1_batch)
        emb_llm_2 = self.encoder_llm_2(data.x_llm_2, data.edge_index_llm_2, data.edge_attr_llm_2, data.x_llm_2_batch)

        emb_llm_1 = F.normalize(emb_llm_1, dim=1)
        emb_llm_2 = F.normalize(emb_llm_2, dim=1)
        return emb_llm_1, emb_llm_2
