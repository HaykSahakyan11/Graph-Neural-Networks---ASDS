import torch.nn as nn
from torch.nn import Dropout
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, Sequential
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

# class GNNModel(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
#         super(GNNModel, self).__init__()
#
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.bn1 = BatchNorm(hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, hidden_channels)
#         self.bn2 = BatchNorm(hidden_channels)
#         self.conv3 = GCNConv(hidden_channels, out_channels)
#         self.dropout = Dropout(p=dropout)
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#
#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = self.dropout(x)
#
#         x = self.conv3(x, edge_index)
#         return x
#
#     def get_embeddings(self, x, edge_index):
#         """Extract node embeddings before final layer."""
#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
#         return x
