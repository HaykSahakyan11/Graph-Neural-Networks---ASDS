import torch
import torch.nn as nn
from torch.nn import Dropout
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, Sequential, GATConv, SAGEConv, DenseGCNConv, global_add_pool
from src.config import CONFIG

config = CONFIG()


class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super(GCNModel, self).__init__()

        # First layer: GCN
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)

        # Second layer: Multi-Head GAT with more heads
        self.conv2 = GATConv(hidden_channels, hidden_channels // 4, heads=4, concat=False)  # Using 8 attention heads
        self.bn2 = BatchNorm(hidden_channels // 4)

        # Third layer: GCN for additional transformation
        self.conv3 = GCNConv(hidden_channels // 4, hidden_channels // 16)
        self.bn3 = BatchNorm(hidden_channels // 16)

        # Fourth layer: GCN for final transformation
        self.conv4 = GCNConv(hidden_channels // 16, out_channels)

        self.dropout = Dropout(p=dropout)

        # Additional Residual Connections
        self.res1 = nn.Linear(in_channels, hidden_channels)
        self.res2 = nn.Linear(hidden_channels, hidden_channels // 4)
        self.res3 = nn.Linear(hidden_channels // 4, hidden_channels // 16)

        # Final Projection Layer
        self.projection = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        x_input = x.clone()  # For initial residual connection

        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = x + self.res1(x_input)  # Residual connection from input

        # Layer 2
        x_input2 = x.clone()
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = x + self.res2(x_input2)  # Residual connection

        # Layer 3
        x_input3 = x.clone()
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = x + self.res3(x_input3)  # Residual connection

        # Layer 4
        x = self.conv4(x, edge_index)
        x = self.projection(x)  # Feature projection

        return x

    def get_embeddings(self, x, edge_index):
        """Extract node embeddings before final layer."""
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

        # SEAL's subgraph embedding approach
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)

        self.conv2 = SAGEConv(hidden_channels, hidden_channels // 2)
        self.bn2 = BatchNorm(hidden_channels // 2)

        self.conv3 = GCNConv(hidden_channels // 2, hidden_channels // 4)
        self.bn3 = BatchNorm(hidden_channels // 4)

        self.conv4 = GCNConv(hidden_channels // 4, out_channels)  # Replaced DenseGCNConv with GCNConv
        self.dropout = Dropout(p=dropout)

        self.readout = global_add_pool  # Readout function for subgraph embeddings
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.ReLU(),
            nn.Linear(out_channels // 2, 1)
        )

    def forward(self, x, edge_index, batch, link_indices):
        """Forward method now only processes batched link indices."""
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
        x = self.readout(x, batch)  # Readout for whole subgraph embedding

        # Extract only link-relevant embeddings
        src_emb = x[link_indices[0]]
        tgt_emb = x[link_indices[1]]

        # Compute dot product prediction
        link_pred = torch.sigmoid((src_emb * tgt_emb).sum(dim=1))
        return link_pred

    def get_embeddings(self, x, edge_index, batch):
        """Extract node embeddings before final layer."""
        if batch is None or batch.numel() == 0 or batch.shape[0] != x.shape[0]:
            batch = torch.arange(x.size(0), device=x.device)

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        return self.readout(x, batch)










# class SEALModel(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
#         super(SEALModel, self).__init__()
#
#         # SEAL's subgraph embedding approach
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.bn1 = BatchNorm(hidden_channels)
#
#         self.conv2 = SAGEConv(hidden_channels, hidden_channels // 2)
#         self.bn2 = BatchNorm(hidden_channels // 2)
#
#         self.conv3 = GCNConv(hidden_channels // 2, hidden_channels // 4)
#         self.bn3 = BatchNorm(hidden_channels // 4)
#
#         self.conv4 = DenseGCNConv(hidden_channels // 4, out_channels)
#         self.dropout = Dropout(p=dropout)
#
#         self.readout = global_add_pool  # Readout function for subgraph embeddings
#         self.mlp = nn.Sequential(
#             nn.Linear(out_channels, out_channels // 2),
#             nn.ReLU(),
#             nn.Linear(out_channels // 2, 1)
#         )
#
#     def forward(self, x, edge_index, batch):
#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.leaky_relu(x)
#         x = self.dropout(x)
#
#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.leaky_relu(x)
#         x = self.dropout(x)
#
#         x = self.conv3(x, edge_index)
#         x = self.bn3(x)
#         x = F.leaky_relu(x)
#         x = self.dropout(x)
#
#         x = self.conv4(x, edge_index)
#         x = self.readout(x, batch)  # Readout for whole subgraph embedding
#         x = self.mlp(x)  # Final MLP layer for link prediction
#         return x
#
#     def get_embeddings(self, x, edge_index, batch):
#         """Extract node embeddings before final layer."""
#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.leaky_relu(x)
#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.leaky_relu(x)
#         return self.readout(x, batch)

# class SEALModel(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
#         super(SEALModel, self).__init__()
#
#         # SEAL's subgraph embedding approach
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.bn1 = BatchNorm(hidden_channels)
#
#         self.conv2 = SAGEConv(hidden_channels, hidden_channels // 2)
#         self.bn2 = BatchNorm(hidden_channels // 2)
#
#         self.conv3 = GCNConv(hidden_channels // 2, hidden_channels // 4)
#         self.bn3 = BatchNorm(hidden_channels // 4)
#
#         self.conv4 = GCNConv(hidden_channels // 4, out_channels)
#         self.dropout = Dropout(p=dropout)
#
#         self.readout = global_add_pool  # Readout function for subgraph embeddings
#         self.mlp = nn.Sequential(
#             nn.Linear(out_channels, out_channels // 2),
#             nn.ReLU(),
#             nn.Linear(out_channels // 2, 1)
#         )
#
#     def forward(self, x, edge_index, batch):
#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.leaky_relu(x)
#         x = self.dropout(x)
#
#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.leaky_relu(x)
#         x = self.dropout(x)
#
#         x = self.conv3(x, edge_index)
#         x = self.bn3(x)
#         x = F.leaky_relu(x)
#         x = self.dropout(x)
#
#         x = self.conv4(x, edge_index)
#         x = self.readout(x, batch)  # Readout for whole subgraph embedding
#         x = self.mlp(x)  # Final MLP layer for link prediction
#         return x
#
#     def get_embeddings(self, x, edge_index, batch):
#         """Extract node embeddings before final MLP layer."""
#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.leaky_relu(x)
#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.leaky_relu(x)
#         x = self.conv3(x, edge_index)
#         x = self.bn3(x)
#         x = F.leaky_relu(x)
#         x = self.conv4(x, edge_index)
#         return self.readout(x, batch)

