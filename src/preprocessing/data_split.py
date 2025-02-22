import torch

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


if __name__ == '__main__':
    edge_file = "path/to/train.txt"
    train_links, train_labels, val_links, val_labels = stratified_split(edge_file)

    print("Train set size:", train_links.shape[1])
    print("Validation set size:", val_links.shape[1])
