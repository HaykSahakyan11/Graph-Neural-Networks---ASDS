import torch
import torch.nn as nn
import torch.optim as optim
from src.datasets.gnn_dataset import ACTORNETWORKData
from src.preprocessing.data_split import stratified_split
from src.config import CONFIG
from src.models.model import GNNModel

from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

config = CONFIG()


def decode_links(link_pairs, node_id_map):
    """Convert mapped node indices back to original IDs."""
    # inverse_map = {v: k for k, v in node_id_map.items()}
    decoded_links = torch.tensor([[node_id_map[src.item()], node_id_map[tgt.item()]] for src, tgt in link_pairs.T]).T
    return decoded_links


def train():
    # Load dataset
    node_feature_file = config.node_features
    edge_file = config.train_data
    dataset = ACTORNETWORKData(node_feature_file, edge_file)

    # Split dataset into train and validation sets
    train_links, train_labels, val_links, val_labels = stratified_split(edge_file)

    # Decode train and validation links to original node IDs
    train_links = decode_links(train_links, dataset.node_id_map)
    val_links = decode_links(val_links, dataset.node_id_map)

    # Initialize model
    model = GNNModel(in_channels=dataset.x.shape[1], hidden_channels=64,
                     out_channels=2)  # Output 1 node for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        node_embeddings = model(dataset.x, dataset.edge_index)

        # Get embeddings for link prediction
        src_emb = node_embeddings[train_links[0]]
        tgt_emb = node_embeddings[train_links[1]]
        link_preds = torch.sigmoid((src_emb * tgt_emb).sum(dim=1))  # Sigmoid for binary classification
        loss = criterion(link_preds, train_labels)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Validation
    model.eval()
    with torch.no_grad():
        src_emb = node_embeddings[val_links[0]]
        tgt_emb = node_embeddings[val_links[1]]
        val_preds = torch.sigmoid((src_emb * tgt_emb).sum(dim=1))
        val_loss = criterion(val_preds, val_labels)

        # Convert predictions to binary labels
        val_preds_bin = (val_preds > 0.5).float()
        accuracy = accuracy_score(val_labels.cpu(), val_preds_bin.cpu())
        auc = roc_auc_score(val_labels.cpu(), val_preds.cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels.cpu(), val_preds_bin.cpu(),
                                                                   average='binary')

        print("Validation Loss:", val_loss.item())
        print(
            f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")


if __name__ == '__main__':
    train()
