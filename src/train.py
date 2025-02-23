import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from src.datasets.gnn_dataset import ACTORNETWORKData
from src.preprocessing.data_split import stratified_split
from src.models.model import GNNModel, GCNModel, SEALModel
from src.config import CONFIG, set_seed

set_seed()
config = CONFIG()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_best_model(model_state, best_acc, best_epoch, model_name):
    model_dir = config.model_dir
    to_save_path = os.path.join(model_dir, f"{model_name}_epoch_{best_epoch}_best_acc_{best_acc:.3f}.pth")
    torch.save(model_state, to_save_path)
    print(f"Best model saved at epoch {best_epoch} with Accuracy: {best_acc:.4f}")


def save_history(history, model_name):
    df = pd.DataFrame(history)
    history_path = os.path.join(config.log_dir, f"{model_name}_history.csv")
    df.to_csv(history_path, index=False)
    print(f"Training history saved at {history_path}")


class Trainer:
    def __init__(self, model_name, file_name=None):
        self.model_name = model_name
        self.file_name = file_name
        self.model = None
        self.hid_channels = None
        self.out_channels = None
        self.dropout = None
        self.lr = None
        self.weight_decay = None
        self.epochs = None

        self.criterion = None
        self.optimizer = None

        self.dataset = None
        self.config = CONFIG()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.set_attributes()

    def set_attributes(self):
        if self.model_name == 'GCN_GAT':
            self.hid_channels = config.train_params['gcn_params']['hidden_channels']
            self.out_channels = config.train_params['gcn_params']['out_channels']
            self.dropout = config.train_params['gcn_params']['dropout']
            self.lr = config.train_params['gcn_params']['lr']
            self.weight_decay = config.train_params['gcn_params']['weight_decay']
            self.epochs = config.train_params['gcn_params']['epochs']
        elif self.model_name == 'SEAL':
            self.hid_channels = config.train_params['seal_params']['hidden_channels']
            self.out_channels = config.train_params['seal_params']['out_channels']
            self.dropout = config.train_params['seal_params']['dropout']
            self.lr = config.train_params['seal_params']['lr']
            self.weight_decay = config.train_params['seal_params']['weight_decay']
            self.epochs = config.train_params['seal_params']['epochs']

    def train(self):
        node_feature_file = self.config.node_features
        edge_file = self.config.train_data
        dataset = ACTORNETWORKData(node_feature_file, edge_file)
        history = []

        if self.model_name == 'GCN_GAT':
            history = self.train_gcn_gat_model(dataset=dataset)
        elif self.model_name == 'SEAL':
            history = self.train_seal_model(dataset=dataset)

        save_history(history, self.model_name)
        return history

    def train_seal_model(self, dataset):
        dataset = dataset.to(self.device)
        in_channels = dataset.x.shape[1]

        train_links, train_labels, val_links, val_labels = stratified_split(dataset.link_pairs, dataset.labels)

        model = SEALModel(
            in_channels=in_channels, hidden_channels=self.hid_channels,
            out_channels=self.out_channels, dropout=self.dropout
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        train_loader = DataLoader(list(zip(train_links.T, train_labels)), batch_size=32, shuffle=True)
        val_loader = DataLoader(list(zip(val_links.T, val_labels)), batch_size=32, shuffle=False)

        best_acc = 0.0
        best_model_state = None
        best_epoch = 0
        history = []

        for epoch in tqdm(range(self.epochs), desc="Training Epochs", initial=1):
            model.train()
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
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

            # Validation after each epoch
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
                precision, recall, f1, _ = precision_recall_fscore_support(all_labels.cpu(), val_preds_bin.cpu(),
                                                                           average='binary')

                print(f"Validation Loss: {val_loss.item():.4f}, Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, "
                      f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

                history.append({
                    'epoch': epoch + 1,
                    'val_loss': val_loss.item(),
                    'accuracy': accuracy,
                    'auc': auc.item(),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })

                # Update best model if current model is better
                if accuracy > best_acc:
                    best_acc = accuracy
                    best_model_state = model.state_dict()
                    best_epoch = epoch + 1

        # Save best model after training completes
        save_best_model(best_model_state, best_acc, best_epoch, model_name="SEAL")
        return history

    def train_gcn_gat_model(self, dataset):
        dataset = dataset.to(self.device)
        in_channels = dataset.x.shape[1]

        train_links, train_labels, val_links, val_labels = stratified_split(dataset.link_pairs, dataset.labels)

        model = GCNModel(
            in_channels=in_channels, hidden_channels=self.hid_channels,
            out_channels=self.out_channels, dropout=self.dropout
        ).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        best_acc = 0.0
        best_model_state = None
        best_epoch = 0
        history = []

        for epoch in tqdm(range(self.epochs), desc="Training Epochs"):
            model.train()
            optimizer.zero_grad()
            node_embeddings = model(dataset.x.to(self.device), dataset.edge_index.to(self.device))

            src_emb = node_embeddings[train_links[0].to(self.device)]
            tgt_emb = node_embeddings[train_links[1].to(self.device)]
            link_preds = torch.sigmoid((src_emb * tgt_emb).sum(dim=1))

            loss = criterion(link_preds, train_labels.to(self.device))
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

            # Validation after each epoch
            model.eval()
            with torch.no_grad():
                src_emb = node_embeddings[val_links[0].to(self.device)]
                tgt_emb = node_embeddings[val_links[1].to(self.device)]
                val_preds = torch.sigmoid((src_emb * tgt_emb).sum(dim=1))
                val_loss = criterion(val_preds, val_labels.to(self.device))

                val_preds_bin = (val_preds > 0.5).float()
                accuracy = accuracy_score(val_labels.cpu(), val_preds_bin.cpu())
                auc = roc_auc_score(val_labels.cpu(), val_preds.cpu())
                precision, recall, f1, _ = precision_recall_fscore_support(
                    val_labels.cpu(), val_preds_bin.cpu(), average='binary')

                print(f"Validation Loss: {val_loss.item():.4f}, Accuracy: {accuracy:.4f}, "
                      f"AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
                      f"F1-score: {f1:.4f}")

                history.append({
                    'epoch': epoch + 1,
                    'val_loss': val_loss.item(),
                    'accuracy': accuracy,
                    'auc': auc.item(),
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })

                if accuracy > best_acc:
                    best_acc = accuracy
                    best_model_state = model.state_dict()
                    best_epoch = epoch + 1

        save_best_model(best_model_state, best_acc, best_epoch, model_name="GCN_GAT")
        return history


if __name__ == '__main__':
    trainer = Trainer(model_name="SEAL")
    # trainer = Trainer(model_name="GCN_GAT")
    history = trainer.train()
    print(history)
