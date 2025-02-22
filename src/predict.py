import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.datasets.gnn_dataset import ACTORNETWORKData
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

config = CONFIG()


def predict_seal_model(
        model_path: str,
        link_pairs_to_predict,
        batch_size: int = 32
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_channels = config.train_params['seal_params']['hidden_channels']
    out_channels = config.train_params['seal_params']['out_channels']
    dropout = config.train_params['seal_params']['dropout']

    # 1) Load your dataset (graph + node features)
    node_feature_file = config.node_features  # e.g. "node_features.csv"
    edge_file = config.train_data  # e.g. "edges.csv"

    # You may need to import or define ACTORNETWORKData in your code
    dataset = ACTORNETWORKData(node_feature_file, edge_file)
    dataset = dataset.to(device)

    # 2) Figure out the input dimension
    in_channels = dataset.x.shape[1]

    # 3) Initialize the SEAL model with the same parameters used for training
    model = SEALModel(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=dropout
    ).to(device)

    # 4) Load the saved weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 5) Convert link_pairs_to_predict to a torch tensor if not already
    if not isinstance(link_pairs_to_predict, torch.Tensor):
        link_pairs_to_predict = torch.tensor(link_pairs_to_predict, dtype=torch.long)

    link_pairs_to_predict = link_pairs_to_predict.to(device)

    # If shape is (2, N), transpose to (N, 2)
    if link_pairs_to_predict.shape[0] == 2:
        link_pairs_to_predict = link_pairs_to_predict.t()

    # Create a dataset & DataLoader for prediction
    # We attach a dummy label (0.0) since we only need the forward pass
    predict_dataset = [(lp, torch.tensor([0.0])) for lp in link_pairs_to_predict]
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)

    all_preds = []

    # 6) Inference loop
    with torch.no_grad():
        for batch_links, _ in predict_loader:
            # batch_links shape: (batch_size, 2)
            # The model expects shape: (2, batch_size), so transpose:
            batch_links = batch_links.t().to(device)
            # Forward pass (you might not need the 'None' if your forward is simpler)
            preds = model(dataset.x.to(device), dataset.edge_index.to(device), None, batch_links)
            all_preds.append(preds)

    # 7) Concatenate all predictions into a single tensor
    all_preds = torch.cat(all_preds, dim=0)

    # Optionally, convert logits to probabilities:
    # all_preds = torch.sigmoid(all_preds)

    return all_preds

if __name__ == "__main__":
    import csv


    # Instantiate your config object
    config = CONFIG()
    test_path = config.test_data
    with open(test_path, "r") as f:
        reader = csv.reader(f)
        test_set = list(reader)
    test_set = [element[0].split(" ") for element in test_set]
    test_set = [[int(element[0]), int(element[1])] for element in test_set]

    print(test_set)

    # # Suppose you have a set of new edges (2 x N or N x 2)
    # new_links = [
    #     [12, 3],
    #     [50, 8],
    #     [30, 33]
    # ]
    # # shape is (3,2) in this example
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    model_dir = config.model_dir
    model_path = os.path.join(model_dir, "SEAL_best_acc_0.806.pth")


    logits = predict_seal_model(
        model_path=model_path,  # The file path where you saved your best model
        link_pairs_to_predict=test_set,
        batch_size=32
    )

    # logits are raw scores. Convert to probabilities if needed:
    probs = torch.sigmoid(logits)

    # If you want binary predictions (0 or 1), threshold at 0.5:
    predictions = (probs > 0.5).float()

    print("Logits:", logits)
    print("Probabilities:", probs)
    print("Binary Predictions:", predictions)
