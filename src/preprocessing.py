import torch
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit

from src.config import CONFIG, set_seed

config = CONFIG()
set_seed()


def stratified_split(link_pairs: torch.Tensor, labels: torch.Tensor):
    """
    Splits the link dataset into training and validation sets while preserving class distribution.

    Args:
        link_pairs (torch.Tensor): A tensor containing node pairs (edges) in the dataset.
        labels (torch.Tensor): A tensor containing labels (1 for existing edges, 0 for non-edges).

    Returns:
        tuple:
            - train_links (torch.LongTensor): Training set link pairs.
            - train_labels (torch.Tensor): Training set labels.
            - val_links (torch.LongTensor): Validation set link pairs.
            - val_labels (torch.Tensor): Validation set labels.
    """
    train_size = config.train_params['train_size']

    link_pairs_np = link_pairs.T.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Perform stratified shuffle split
    strat_split = StratifiedShuffleSplit(n_splits=1, train_size=train_size)
    train_idx, val_idx = next(strat_split.split(link_pairs_np, labels_np))

    train_links = torch.tensor(link_pairs_np[train_idx].T, dtype=torch.long)
    train_labels = torch.tensor(labels_np[train_idx], dtype=torch.float)
    val_links = torch.tensor(link_pairs_np[val_idx].T, dtype=torch.long)
    val_labels = torch.tensor(labels_np[val_idx], dtype=torch.float)

    return train_links, train_labels, val_links, val_labels


def load_node_features(filepath):
    df = pd.read_csv(filepath, header=None)
    node_indices = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values.astype(float)
    x = torch.tensor(features, dtype=torch.float)

    node_id_map = {old_id: new_id for new_id, old_id in enumerate(node_indices)}
    return node_id_map, x


def load_graph_edges(filepath, node_id_map):
    df = pd.read_csv(filepath, sep='\s+', header=None, names=['source', 'target', 'label'])
    df = df[df['label'] == 1]

    df = df[df['source'].isin(node_id_map) & df['target'].isin(node_id_map)]
    df['source'] = df['source'].map(node_id_map)
    df['target'] = df['target'].map(node_id_map)

    edge_index = torch.tensor(df[['source', 'target']].values.T, dtype=torch.long)
    return edge_index


def load_link_labels(filepath, node_id_map):
    df = pd.read_csv(filepath, sep='\s+', header=None, names=['source', 'target', 'label'])

    df = df[df['source'].isin(node_id_map) & df['target'].isin(node_id_map)]
    df['source'] = df['source'].map(node_id_map)
    df['target'] = df['target'].map(node_id_map)

    link_pairs = torch.tensor(df[['source', 'target']].values.T, dtype=torch.long)
    labels = torch.tensor(df['label'].values, dtype=torch.float)
    return link_pairs, labels


def load_test_links(filepath, node_id_map):
    df = pd.read_csv(filepath, sep='\s+', header=None, names=['source', 'target'])

    df = df[df['source'].isin(node_id_map) & df['target'].isin(node_id_map)]
    df['source'] = df['source'].map(node_id_map)
    df['target'] = df['target'].map(node_id_map)

    test_links = torch.tensor(df.values.T, dtype=torch.long)
    return test_links

