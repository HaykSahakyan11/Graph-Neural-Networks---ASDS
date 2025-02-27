import torch
import pandas as pd
import numpy as np
import networkx as nx

from torch_geometric.data import Data as Data
from torch_geometric.utils import degree

from src.preprocessing import load_node_features, load_graph_edges, load_link_labels
from src.config import CONFIG

config = CONFIG()


class ACTORNETWORKData(Data):
    def __init__(self, node_feature_file, edge_file):
        node_id_map, x = load_node_features(node_feature_file)

        edge_index = load_graph_edges(edge_file, node_id_map)

        link_pairs, labels = load_link_labels(edge_file, node_id_map)

        super().__init__(x=x, edge_index=edge_index)

        self.link_pairs = link_pairs
        self.labels = labels
        self.node_id_map = node_id_map

    def get_node_features(self, node_id):
        new_id = self.node_id_map.get(node_id)
        if new_id is None:
            return torch.zeros(self.x.shape[1])
        return self.x[new_id]

    def get_edge_list(self):
        return self.edge_index.T.tolist()

    def get_labels(self):
        return self.labels


def get_actors_network_graph():
    node_feature_file = config.node_features
    edge_file = config.train_data
    dataset = ACTORNETWORKData(node_feature_file, edge_file)
    return dataset


class ACTORNETWORKData_v2(Data):
    def __init__(self, node_feature_file, edge_file):
        node_df = pd.read_csv(node_feature_file, header=None)
        node_features = node_df.iloc[:, 1:].values.astype(np.float32)

        num_nodes = node_df.iloc[:, 0].max() + 1

        x = torch.zeros((num_nodes, node_features.shape[1]), dtype=torch.float)
        x[node_df.iloc[:, 0].values] = torch.tensor(node_features, dtype=torch.float)

        train_df = pd.read_csv(edge_file, sep=' ', header=None, names=['src', 'dst', 'label'])

        positive_edges = train_df[train_df['label'] == 1][['src', 'dst']].values
        edge_index_np = np.concatenate([positive_edges, positive_edges[:, ::-1]], axis=0)
        edge_index = torch.tensor(edge_index_np.T, dtype=torch.long)

        deg = degree(edge_index[0], num_nodes=num_nodes)
        deg = deg.unsqueeze(1)

        G = nx.Graph()
        G.add_edges_from(positive_edges)

        pagerank_scores = nx.pagerank(G)
        pagerank_tensor = torch.zeros((num_nodes, 1), dtype=torch.float)
        for node, score in pagerank_scores.items():
            if node < num_nodes:  # Ensure node index is within bounds
                pagerank_tensor[node] = score

        # # Compute Clustering Coefficient
        # clustering_coeffs = nx.clustering(G)
        # clustering_tensor = torch.zeros((num_nodes, 1), dtype=torch.float)
        # for node, score in clustering_coeffs.items():
        #     if node < num_nodes:
        #         clustering_tensor[node] = score
        #
        # # Compute Betweenness Centrality
        # betweenness_centrality = nx.betweenness_centrality(G)
        # betweenness_tensor = torch.zeros((num_nodes, 1), dtype=torch.float)
        # for node, score in betweenness_centrality.items():
        #     if node < num_nodes:
        #         betweenness_tensor[node] = score
        #
        # # Compute Eigenvector Centrality
        # try:
        #     eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        #     eigenvector_tensor = torch.zeros((num_nodes, 1), dtype=torch.float)
        #     for node, score in eigenvector_centrality.items():
        #         if node < num_nodes:
        #             eigenvector_tensor[node] = score
        # except:
        #     eigenvector_tensor = torch.zeros((num_nodes, 1), dtype=torch.float)  # Handle convergence issues
        #
        # # Compute Jaccard Coefficient (for node pairs)
        # jaccard_coeff = dict(nx.jaccard_coefficient(G, positive_edges))
        # jaccard_tensor = torch.zeros((num_nodes, 1), dtype=torch.float)
        # for (u, v), score in jaccard_coeff.items():
        #     if u < num_nodes:
        #         jaccard_tensor[u] = score
        #     if v < num_nodes:
        #         jaccard_tensor[v] = score
        #
        # # Compute Adamic-Adar Index (for node pairs)
        # adamic_adar_index = dict(nx.adamic_adar_index(G, positive_edges))
        # adamic_tensor = torch.zeros((num_nodes, 1), dtype=torch.float)
        # for (u, v), score in adamic_adar_index.items():
        #     if u < num_nodes:
        #         adamic_tensor[u] = score
        #     if v < num_nodes:
        #         adamic_tensor[v] = score
        #
        # # Compute Resource Allocation Index (for node pairs)
        # resource_alloc_index = dict(nx.resource_allocation_index(G, positive_edges))
        # resource_tensor = torch.zeros((num_nodes, 1), dtype=torch.float)
        # for (u, v), score in resource_alloc_index.items():
        #     if u < num_nodes:
        #         resource_tensor[u] = score
        #     if v < num_nodes:
        #         resource_tensor[v] = score

        # x_aug = torch.cat([x, deg], dim=1)
        # x_aug = torch.cat([
        #     x, deg, pagerank_tensor, clustering_tensor,
        #     betweenness_tensor, eigenvector_tensor, jaccard_tensor,
        #     adamic_tensor, resource_tensor
        # ], dim=1)
        x_aug = torch.cat([x, deg, pagerank_tensor], dim=1)

        super().__init__(x=x_aug, edge_index=edge_index)

        self.train_edge_pairs = torch.tensor(train_df[['src', 'dst']].values.T, dtype=torch.long)
        self.train_labels = torch.tensor(train_df['label'].values, dtype=torch.float)

    def get_node_features(self, node_id):
        new_id = self.node_id_map.get(node_id)
        if new_id is None:
            return torch.zeros(self.x.shape[1])
        return self.x[new_id]

    def get_edge_list(self):
        return self.edge_index.T.tolist()

    def get_labels(self):
        return self.train_labels


if __name__ == '__main__':
    dataset = get_actors_network_graph()

    print("Dataset loaded successfully!")
    print("Graph structure:", dataset.edge_index.shape)
    print("Number of nodes:", dataset.x.shape[0])
    print("Number of edges:", dataset.edge_index.shape[1])
    print("Number of link pairs:", dataset.link_pairs.shape[1])
    # node_feature_file = config.node_features
    # edge_file = config.train_data
    # dataset = ACTORNETWORKData(node_feature_file, edge_file)
    #
    # print("Dataset loaded successfully!")
    # print("Graph structure:", dataset.edge_index.shape)
    # print("Number of nodes:", dataset.x.shape[0])
    # print("Number of edges:", dataset.edge_index.shape[1])
    # print("Number of link pairs:", dataset.link_pairs.shape[1])
