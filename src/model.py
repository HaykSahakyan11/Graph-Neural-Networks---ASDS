import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Dropout
from torch_geometric.nn import GCNConv, BatchNorm, GATConv, SAGEConv, global_add_pool, global_mean_pool
from src.config import CONFIG

config = CONFIG()


class GCN_Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN_Model, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.lin = torch.nn.Linear(2 * out_channels, 1)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode_mlp(self, z, edge_pairs):
        src, dst = edge_pairs
        h = torch.cat([z[src], z[dst]], dim=1)
        return torch.sigmoid(self.lin(h)).view(-1)

    def forward(self, data, edge_pairs):
        z = self.encode(data.x, data.edge_index)
        return self.decode_mlp(z, edge_pairs)


class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super(GCNModel, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)

        self.conv2 = GATConv(hidden_channels, hidden_channels // 4, heads=4, concat=False)  # Using 8 attention heads
        self.bn2 = BatchNorm(hidden_channels // 4)

        self.conv3 = GCNConv(hidden_channels // 4, hidden_channels // 16)
        self.bn3 = BatchNorm(hidden_channels // 16)

        self.conv4 = GCNConv(hidden_channels // 16, out_channels)

        self.dropout = Dropout(p=dropout)

        self.res1 = nn.Linear(in_channels, hidden_channels)
        self.res2 = nn.Linear(hidden_channels, hidden_channels // 4)
        self.res3 = nn.Linear(hidden_channels // 4, hidden_channels // 16)

        self.projection = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        x_input = x.clone()

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = x + self.res1(x_input)

        x_input2 = x.clone()
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = x + self.res2(x_input2)

        x_input3 = x.clone()
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = x + self.res3(x_input3)

        # Layer 4
        x = self.conv4(x, edge_index)
        x = self.projection(x)

        return x

    def get_embeddings(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        return x


class SEALModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super(SEALModel, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)

        self.conv2 = SAGEConv(hidden_channels, hidden_channels // 2)
        self.bn2 = BatchNorm(hidden_channels // 2)

        self.conv3 = GCNConv(hidden_channels // 2, hidden_channels // 4)
        self.bn3 = BatchNorm(hidden_channels // 4)

        self.conv4 = GCNConv(hidden_channels // 4, out_channels)
        self.dropout = Dropout(p=dropout)

        self.readout = global_add_pool
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.ReLU(),
            nn.Linear(out_channels // 2, 1)
        )

    def forward(self, x, edge_index, batch, link_indices):
        if batch is None or batch.numel() == 0 or batch.shape[0] != x.shape[0]:
            batch = torch.arange(x.size(0), device=x.device)  # Assign unique batch indices per node

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x, edge_index)
        x = self.readout(x, batch)

        src_emb = x[link_indices[0]]
        tgt_emb = x[link_indices[1]]

        link_pred = torch.sigmoid((src_emb * tgt_emb).sum(dim=1))
        return link_pred

    def get_embeddings(self, x, edge_index, batch):
        if batch is None or batch.numel() == 0 or batch.shape[0] != x.shape[0]:
            batch = torch.arange(x.size(0), device=x.device)

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        return self.readout(x, batch)


class ImprovedSEALModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super(ImprovedSEALModel, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.res1 = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None

        hidden_channels2 = hidden_channels // 2
        self.conv2 = GCNConv(hidden_channels, hidden_channels2)
        self.ln2 = nn.LayerNorm(hidden_channels2)
        self.res2 = nn.Linear(hidden_channels, hidden_channels2)

        hidden_channels3 = hidden_channels // 4
        self.conv3 = GCNConv(hidden_channels2, hidden_channels3)
        self.ln3 = nn.LayerNorm(hidden_channels3)
        self.res3 = nn.Linear(hidden_channels2, hidden_channels3)

        self.conv4 = GCNConv(hidden_channels3, out_channels)
        self.ln4 = nn.LayerNorm(out_channels)
        self.res4 = nn.Linear(hidden_channels3, out_channels)

        self.dropout = nn.Dropout(p=dropout)

        self.readout = global_mean_pool

        self.link_mlp = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(out_channels, out_channels // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(out_channels // 2, 1)
        )

    def forward(self, x, edge_index, batch, link_indices):
        if batch is None or batch.numel() == 0 or batch.shape[0] != x.shape[0]:
            batch = torch.arange(x.size(0), device=x.device)

        x1 = self.conv1(x, edge_index)
        res1 = self.res1(x) if self.res1 is not None else x
        x1 = x1 + res1
        x1 = self.ln1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)

        x2 = self.conv2(x1, edge_index)
        res2 = self.res2(x1)
        x2 = x2 + res2
        x2 = self.ln2(x2)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)

        x3 = self.conv3(x2, edge_index)
        res3 = self.res3(x2)
        x3 = x3 + res3
        x3 = self.ln3(x3)
        x3 = F.relu(x3)
        x3 = self.dropout(x3)

        x4 = self.conv4(x3, edge_index)
        res4 = self.res4(x3)
        x4 = x4 + res4
        x4 = self.ln4(x4)
        x4 = F.relu(x4)
        x4 = self.dropout(x4)

        graph_emb = self.readout(x4, batch)

        src_emb = graph_emb[link_indices[0]]
        tgt_emb = graph_emb[link_indices[1]]

        link_features = torch.cat([src_emb, tgt_emb], dim=1)
        link_pred = torch.sigmoid(self.link_mlp(link_features))
        return link_pred.squeeze()

    def get_embeddings(self, x, edge_index, batch):
        if batch is None or batch.numel() == 0 or batch.shape[0] != x.shape[0]:
            batch = torch.arange(x.size(0), device=x.device)

        x1 = self.conv1(x, edge_index)
        res1 = self.res1(x) if self.res1 is not None else x
        x1 = x1 + res1
        x1 = self.ln1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1, edge_index)
        res2 = self.res2(x1)
        x2 = x2 + res2
        x2 = self.ln2(x2)
        x2 = F.relu(x2)

        x3 = self.conv3(x2, edge_index)
        res3 = self.res3(x2)
        x3 = x3 + res3
        x3 = self.ln3(x3)
        x3 = F.relu(x3)

        x4 = self.conv4(x3, edge_index)
        res4 = self.res4(x3)
        x4 = x4 + res4
        x4 = self.ln4(x4)
        x4 = F.relu(x4)

        graph_emb = self.readout(x4, batch)
        return graph_emb


class EvenBetterSEALModel(nn.Module):

    def __init__(self, in_channels, hidden_channels=256, out_channels=64, dropout=0.3):
        super(EvenBetterSEALModel, self).__init__()

        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.res1 = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None

        hidden2 = hidden_channels // 2
        self.gat2 = GATConv(hidden_channels, hidden2, heads=4, concat=False)
        self.ln2 = nn.LayerNorm(hidden2)
        self.res2 = nn.Linear(hidden_channels, hidden2)

        hidden3 = hidden2 // 2 if hidden2 >= 2 else 1
        self.sage3 = SAGEConv(hidden2, hidden3)
        self.ln3 = nn.LayerNorm(hidden3)
        self.res3 = nn.Linear(hidden2, hidden3)

        self.gc4 = GCNConv(hidden3, out_channels)
        self.ln4 = nn.LayerNorm(out_channels)
        self.res4 = nn.Linear(hidden3, out_channels) if hidden3 != out_channels else None

        self.dropout = nn.Dropout(p=dropout)

        self.readout_linear = nn.Linear(out_channels, 1)  # used to produce attention/gates

        self.link_mlp = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, 1)
        )

    def forward(self, x, edge_index, batch, link_indices):
        if batch is None or batch.numel() == 0 or batch.shape[0] != x.shape[0]:
            batch = torch.arange(x.size(0), device=x.device)

        h1 = self.gcn1(x, edge_index)
        if self.res1 is not None:
            h1 = h1 + self.res1(x)
        h1 = self.ln1(h1)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)

        h2 = self.gat2(h1, edge_index)
        h2 = h2 + self.res2(h1)
        h2 = self.ln2(h2)
        h2 = F.relu(h2)
        h2 = self.dropout(h2)

        h3 = self.sage3(h2, edge_index)
        h3 = h3 + self.res3(h2)
        h3 = self.ln3(h3)
        h3 = F.relu(h3)
        h3 = self.dropout(h3)

        h4 = self.gc4(h3, edge_index)
        if self.res4 is not None:
            h4 = h4 + self.res4(h3)
        h4 = self.ln4(h4)
        h4 = F.relu(h4)
        h4 = self.dropout(h4)

        gate_scores = torch.sigmoid(self.readout_linear(h4))  # shape [num_nodes, 1]
        gated = h4 * gate_scores  # shape [num_nodes, out_channels]
        gated_sum = torch.zeros(batch.max().item() + 1, h4.size(1), device=h4.device)
        denom_sum = torch.zeros(batch.max().item() + 1, 1, device=h4.device)
        gated_sum = gated_sum.scatter_add_(0, batch.unsqueeze(-1).expand_as(gated), gated)
        denom_sum = denom_sum.scatter_add_(0, batch.unsqueeze(-1), gate_scores)
        denom_sum = denom_sum + 1e-8
        graph_emb = gated_sum / denom_sum

        src_emb = graph_emb[link_indices[0]]
        dst_emb = graph_emb[link_indices[1]]
        link_features = torch.cat([src_emb, dst_emb], dim=1)
        logits = self.link_mlp(link_features)
        link_prob = torch.sigmoid(logits)
        return link_prob.squeeze(-1)

    def get_embeddings(self, x, edge_index, batch):
        if batch is None or batch.numel() == 0 or batch.shape[0] != x.shape[0]:
            batch = torch.arange(x.size(0), device=x.device)

        h1 = self.gcn1(x, edge_index)
        if self.res1 is not None:
            h1 = h1 + self.res1(x)
        h1 = self.ln1(h1)
        h1 = F.relu(h1)

        h2 = self.gat2(h1, edge_index)
        h2 = h2 + self.res2(h1)
        h2 = self.ln2(h2)
        h2 = F.relu(h2)

        h3 = self.sage3(h2, edge_index)
        h3 = h3 + self.res3(h2)
        h3 = self.ln3(h3)
        h3 = F.relu(h3)

        h4 = self.gc4(h3, edge_index)
        if self.res4 is not None:
            h4 = h4 + self.res4(h3)
        h4 = self.ln4(h4)
        h4 = F.relu(h4)

        return h4
