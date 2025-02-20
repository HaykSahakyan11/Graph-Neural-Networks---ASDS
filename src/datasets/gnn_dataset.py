import pandas as pd
import torch

from torch_geometric.data import Data as Data
from src.config import CONFIG

config = CONFIG()


def load_node_features(filepath):
    """
    Reads the node_information.csv file and returns:
      - node index mapping (dict {original_id: new_id})
      - node feature tensor (torch.FloatTensor)
    """
    df = pd.read_csv(filepath, header=None)
    node_indices = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values.astype(float)
    x = torch.tensor(features, dtype=torch.float)

    # Create mapping from original IDs to new indices
    node_id_map = {old_id: new_id for new_id, old_id in enumerate(node_indices)}
    return node_id_map, x


def load_graph_edges(filepath, node_id_map):
    """
    Reads the train.txt file and constructs the graph structure.
    Node indices are remapped to be continuous.
    """
    df = pd.read_csv(filepath, sep='\s+', header=None, names=['source', 'target', 'label'])
    df = df[df['label'] == 1]

    # Map original node IDs to new indices
    df = df[df['source'].isin(node_id_map) & df['target'].isin(node_id_map)]
    df['source'] = df['source'].map(node_id_map)
    df['target'] = df['target'].map(node_id_map)

    edge_index = torch.tensor(df[['source', 'target']].values.T, dtype=torch.long)
    return edge_index


def load_link_labels(filepath, node_id_map):
    """
    Reads the train.txt file and loads all link pairs along with their labels.
    Node indices are remapped to be continuous.
    """
    df = pd.read_csv(filepath, sep='\s+', header=None, names=['source', 'target', 'label'])

    # Map original node IDs to new indices
    df = df[df['source'].isin(node_id_map) & df['target'].isin(node_id_map)]
    df['source'] = df['source'].map(node_id_map)
    df['target'] = df['target'].map(node_id_map)

    link_pairs = torch.tensor(df[['source', 'target']].values.T, dtype=torch.long)
    labels = torch.tensor(df['label'].values, dtype=torch.float)
    return link_pairs, labels


def load_test_links(filepath, node_id_map):
    """
    Reads the test.txt file, which contains unlabeled link pairs.
    Node indices are remapped to be continuous.
    """
    df = pd.read_csv(filepath, sep='\s+', header=None, names=['source', 'target'])

    # Map original node IDs to new indices
    df = df[df['source'].isin(node_id_map) & df['target'].isin(node_id_map)]
    df['source'] = df['source'].map(node_id_map)
    df['target'] = df['target'].map(node_id_map)

    test_links = torch.tensor(df.values.T, dtype=torch.long)
    return test_links


class ACTORNETWORKData(Data):
    def __init__(self, node_feature_file, edge_file):
        """Custom data class for GNN link prediction with reindexed nodes."""

        # Load node features and ID mapping
        node_id_map, x = load_node_features(node_feature_file)

        # Load graph edges with reindexed node IDs
        edge_index = load_graph_edges(edge_file, node_id_map)

        # Load link pairs and labels with reindexed node IDs
        link_pairs, labels = load_link_labels(edge_file, node_id_map)

        # Initialize Data object with graph structure only
        super().__init__(x=x, edge_index=edge_index)

        # Store link pairs separately for link prediction tasks
        self.link_pairs = link_pairs
        self.labels = labels
        self.node_id_map = node_id_map

    def get_node_features(self, node_id):
        """Return the features of a single node."""
        new_id = self.node_id_map.get(node_id)
        if new_id is None:
            return torch.zeros(self.x.shape[1])
        return self.x[new_id]

    def get_edge_list(self):
        """Return the list of edges in the graph."""
        return self.edge_index.T.tolist()

    def get_labels(self):
        """Return the labels for the link pairs."""
        return self.labels


if __name__ == '__main__':
    node_feature_file = config.node_features
    edge_file = config.train_data
    dataset = ACTORNETWORKData(node_feature_file, edge_file)

    print("Dataset loaded successfully!")
    print("Graph structure:", dataset.edge_index.shape)
    print("Number of nodes:", dataset.x.shape[0])
    print("Number of edges:", dataset.edge_index.shape[1])
    print("Number of link pairs:", dataset.link_pairs.shape[1])
