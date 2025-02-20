import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from src.datasets.gnn_dataset import ACTORNETWORKData
from src.preprocessing.data_split import stratified_split
from src.models.model import GNNModel, GCNModel, SEALModel
from src.config import CONFIG, set_seed

set_seed()
config = CONFIG()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_best_model(model, val_acc, best_acc, model_name):
    """Save model if it has the best validation accuracy so far."""
    if val_acc > best_acc:
        best_acc = val_acc
        model_dir = config.model_dir
        to_save_path = os.path.join(model_dir, f"{model_name}_best_acc_{val_acc:.3f}.pth")
        torch.save(model.state_dict(), to_save_path)
        print(f"Best model saved with Accuracy: {val_acc:.4f}")
    return best_acc

def train(model_name):
    if model_name == 'GCN_GAT':
        train_gcn_gat_model()
    elif model_name == 'SEAL':
        train_seal_model()

def train_gcn_gat_model():
    # Load dataset
    node_feature_file = config.node_features
    edge_file = config.train_data
    dataset = ACTORNETWORKData(node_feature_file, edge_file).to(device)
    in_channels = dataset.x.shape[1]

    # Split dataset into train and validation sets
    train_links, train_labels, val_links, val_labels = stratified_split(dataset.link_pairs, dataset.labels)

    # Initialize model
    model = GCNModel(in_channels=in_channels, hidden_channels=256, out_channels=2, dropout=0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0  # Track best validation accuracy

    # Training loop
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        node_embeddings = model(dataset.x.to(device), dataset.edge_index.to(device))

        # Get embeddings for link prediction
        src_emb = node_embeddings[train_links[0].to(device)]
        tgt_emb = node_embeddings[train_links[1].to(device)]
        link_preds = torch.sigmoid((src_emb * tgt_emb).sum(dim=1))

        loss = criterion(link_preds, train_labels.to(device))
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Validation
    model.eval()
    with torch.no_grad():
        src_emb = node_embeddings[val_links[0].to(device)]
        tgt_emb = node_embeddings[val_links[1].to(device)]
        val_preds = torch.sigmoid((src_emb * tgt_emb).sum(dim=1))
        val_loss = criterion(val_preds, val_labels.to(device))

        val_preds_bin = (val_preds > 0.5).float()
        accuracy = accuracy_score(val_labels.cpu(), val_preds_bin.cpu())
        auc = roc_auc_score(val_labels.cpu(), val_preds.cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels.cpu(), val_preds_bin.cpu(), average='binary')

        print("Validation Loss:", val_loss.item())
        print(f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

        # Save best model based on validation accuracy
        best_acc = save_best_model(model, accuracy, best_acc, model_name="GCN_GAT")


def train_seal_model():
    # Load dataset
    node_feature_file = config.node_features
    edge_file = config.train_data
    dataset = ACTORNETWORKData(node_feature_file, edge_file).to(device)

    # Split dataset into train and validation sets
    train_links, train_labels, val_links, val_labels = stratified_split(dataset.link_pairs, dataset.labels)

    # Initialize model
    model = SEALModel(in_channels=dataset.x.shape[1], hidden_channels=128, out_channels=64, dropout=0.3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss()

    # DataLoader for batching
    train_loader = DataLoader(list(zip(train_links.T, train_labels)), batch_size=32, shuffle=True)
    val_loader = DataLoader(list(zip(val_links.T, val_labels)), batch_size=32, shuffle=False)

    best_acc = 0.0  # Track best validation accuracy

    # Training loop
    model.train()
    for epoch in range(50):
        total_loss = 0
        for batch_links, batch_labels in train_loader:
            optimizer.zero_grad()
            batch_links = batch_links.t().to(device)
            batch_labels = batch_labels.to(device)
            link_preds = model(dataset.x.to(device), dataset.edge_index.to(device), None, batch_links)
            loss = criterion(link_preds, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_links, batch_labels in val_loader:
            batch_links = batch_links.t().to(device)
            batch_labels = batch_labels.to(device)
            val_preds = model(dataset.x.to(device), dataset.edge_index.to(device), None, batch_links)
            all_preds.append(val_preds)
            all_labels.append(batch_labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        val_loss = criterion(all_preds, all_labels)

        val_preds_bin = (all_preds > 0.5).float()
        accuracy = accuracy_score(all_labels.cpu(), val_preds_bin.cpu())
        auc = roc_auc_score(all_labels.cpu(), all_preds.cpu())
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels.cpu(), val_preds_bin.cpu(), average='binary')

        print("Validation Loss:", val_loss.item())
        print(f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

        # Save best model based on validation accuracy
        best_acc = save_best_model(model, accuracy, best_acc, model_name="SEAL")

if __name__ == '__main__':
    train(model_name="SEAL")
    # train(model_name="GCN_GAT")
