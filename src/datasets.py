import torch

from torch_geometric.data import Data as Data
from preprocessing import load_node_features, load_graph_edges, load_link_labels
from config import CONFIG

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
