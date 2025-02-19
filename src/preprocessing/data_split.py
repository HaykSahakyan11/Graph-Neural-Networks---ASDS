import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from src.datasets.gnn_dataset import load_link_labels


def stratified_split(edge_file, train_ratio=0.85, random_state=42):
    """
    Splits the link dataset into train and validation sets while preserving class distribution.

    Args:
        edge_file (str): Path to the dataset containing links.
        train_ratio (float): Percentage of data to be used for training.
        random_state (int): Random seed for reproducibility.

    Returns:
        train_links (torch.LongTensor): Training link pairs.
        train_labels (torch.Tensor): Training labels.
        val_links (torch.LongTensor): Validation link pairs.
        val_labels (torch.Tensor): Validation labels.
    """

    # Load all link pairs and labels
    link_pairs, labels = load_link_labels(edge_file)

    # Convert tensors to numpy for StratifiedShuffleSplit
    link_pairs_np = link_pairs.T.numpy()
    labels_np = labels.numpy()

    # Perform stratified shuffle split
    strat_split = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_state)
    train_idx, val_idx = next(strat_split.split(link_pairs_np, labels_np))

    # Convert back to tensors
    train_links = torch.tensor(link_pairs_np[train_idx].T, dtype=torch.long)
    train_labels = torch.tensor(labels_np[train_idx], dtype=torch.float)
    val_links = torch.tensor(link_pairs_np[val_idx].T, dtype=torch.long)
    val_labels = torch.tensor(labels_np[val_idx], dtype=torch.float)

    return train_links, train_labels, val_links, val_labels


if __name__ == '__main__':
    edge_file = "path/to/train.txt"  # Update this path as needed
    train_links, train_labels, val_links, val_labels = stratified_split(edge_file)

    print("Train set size:", train_links.shape[1])
    print("Validation set size:", val_links.shape[1])
