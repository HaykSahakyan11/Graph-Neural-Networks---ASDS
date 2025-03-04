import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

from src.datasets import ACTORNETWORKData, ACTORNETWORKData_v2
from src.preprocessing import stratified_split
from src.model import GCNGATModel, SEALModel, ImprovedSEALModel, SageConv_Model
from src.config import CONFIG, set_seed

set_seed()
config = CONFIG()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_best_model(model_state, best_acc, best_epoch, model_name):
    model_dir = config.model_dir
    if model_name == "SageConv_Model":
        to_save_path = os.path.join(model_dir, f"{model_name}_epoch_{best_epoch}_best_loss_{best_acc:.3f}.pth")
    else:
        to_save_path = os.path.join(model_dir, f"{model_name}_epoch_{best_epoch}_best_acc_{best_acc:.3f}.pth")
    torch.save(model_state, to_save_path)
    if model_name == "SageConv_Model":
        output_text = f"Best model saved at epoch {best_epoch} with Loss: {best_acc:.4f}"
    else:
        output_text = f"Best model saved at epoch {best_epoch} with Accuracy: {best_acc:.4f}"
    print(output_text)


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
        elif self.model_name == 'SageConv_Model':
            self.hid_channels = config.train_params['SageConv_Model']['hidden_channels']
            self.out_channels = config.train_params['SageConv_Model']['out_channels']
            self.lr = config.train_params['SageConv_Model']['lr']
            self.weight_decay = config.train_params['SageConv_Model']['weight_decay']
            self.epochs = config.train_params['SageConv_Model']['epochs']

    def train(self):
        node_feature_file = self.config.node_features
        edge_file = self.config.train_data
        dataset = None
        if self.model_name == 'SageConv_Model':
            dataset = ACTORNETWORKData_v2(node_feature_file, edge_file)
        else:
            dataset = ACTORNETWORKData(node_feature_file, edge_file)
        history = []

        if self.model_name == 'GCN_GAT':
            history = self.train_gcn_gat_model(dataset=dataset)
        elif self.model_name == 'SEAL':
            history = self.train_seal_model(dataset=dataset)
        elif self.model_name == 'SageConv_Model':
            history = self.train_sage_conv_model(dataset=dataset)

        save_history(history, self.model_name)
        return history

    def train_sage_conv_model(self, dataset):
        dataset = dataset.to(self.device)
        in_channels = dataset.x.shape[1]

        train_labels = dataset.train_labels
        train_links = dataset.train_edge_pairs

        train_links, train_labels, val_links, val_labels = stratified_split(train_links, train_labels)
        train_labels = train_labels.to(self.device)
        train_links = train_links.to(self.device)
        val_links, val_labels = val_links.to(self.device), val_labels.to(self.device)

        model = SageConv_Model(
            in_channels=in_channels, hidden_channels=self.hid_channels,
            out_channels=self.out_channels
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        history = []
        best_acc = 0.0
        best_model_state = None
        best_epoch = 0
        min_loss = 100.0

        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            pred = model(dataset.to(self.device), train_links.to(self.device))
            loss = F.binary_cross_entropy(pred, train_labels)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                with torch.no_grad():
                    pred_np = pred.detach().cpu().numpy()
                    auc = roc_auc_score(train_labels.cpu().numpy(), pred_np)
                print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f} | AUC: {auc:.4f}')
            # if loss < min_loss:
            #     min_loss = loss
            #     best_model_state = model.state_dict()
            #     best_epoch = epoch + 1
        # save_best_model(best_model_state, min_loss, best_epoch, model_name="SageConv_Model")

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                preds = model(dataset.to(self.device), val_links.to(self.device))
                all_preds.append(preds)
                all_labels.append(val_labels)

                all_preds = torch.cat(all_preds)
                all_labels = torch.cat(all_labels)
                val_loss = F.binary_cross_entropy(all_preds, all_labels)

                val_preds_bin = (all_preds > 0.3).float()
                accuracy = accuracy_score(all_labels.cpu(), val_preds_bin.cpu())
                auc = roc_auc_score(all_labels.cpu(), all_preds.cpu())
                precision, recall, f1, _ = precision_recall_fscore_support(
                    all_labels.cpu(), val_preds_bin.cpu(), average='binary')

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

                if min_loss > val_loss.item():
                    best_acc = accuracy
                    min_loss = val_loss.item()
                    best_model_state = model.state_dict()
                    best_epoch = epoch + 1

        save_best_model(best_model_state, best_acc, best_epoch, model_name="SageConv_Model")
        return history

    def train_seal_model(self, dataset):

        dataset = dataset.to(self.device)
        in_channels = dataset.x.shape[1]

        train_links, train_labels, val_links, val_labels = stratified_split(dataset.link_pairs, dataset.labels)

        model = SEALModel(
            # model = ImprovedSEALModel(
            # model = EvenBetterSEALModel(
            in_channels=in_channels, hidden_channels=self.hid_channels,
            out_channels=self.out_channels, dropout=self.dropout
        ).to(self.device)
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
                batch_links = batch_links.t().to(self.device)
                batch_labels = batch_labels.to(self.device)
                link_preds = model(dataset.x.to(self.device), dataset.edge_index.to(self.device), None, batch_links)
                loss = criterion(link_preds, batch_labels)
                # loss = F.binary_cross_entropy(link_preds, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

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
                # val_loss = F.binary_cross_entropy(all_preds, all_labels)

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

                if accuracy > best_acc:
                    best_acc = accuracy
                    best_model_state = model.state_dict()
                    best_epoch = epoch + 1

        save_best_model(best_model_state, best_acc, best_epoch, model_name="SEAL")
        return history

    def train_gcn_gat_model(self, dataset):
        dataset = dataset.to(self.device)
        in_channels = dataset.x.shape[1]

        train_links, train_labels, val_links, val_labels = stratified_split(dataset.link_pairs, dataset.labels)

        model = GCNGATModel(
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
    # trainer = Trainer(model_name="SEAL")
    # trainer = Trainer(model_name="GCN_GAT")
    trainer = Trainer(model_name="SageConv_Model")
    history = trainer.train()
    print(history)
