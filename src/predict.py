import os
import csv
import torch
import numpy as np
import pandas as pd

from torch_geometric.loader import DataLoader

from src.datasets import get_actors_network_graph, ACTORNETWORKData_v2
from src.preprocessing import load_test_links
from src.model import SEALModel, GCN_Model
from src.config import CONFIG, set_seed

config = CONFIG()
set_seed()

def predict_gcn_model(model_path=None, file_path=None, link_pairs_to_predict=None):
    hidden_channels = config.train_params['GCN_Model']['hidden_channels']
    out_channels = config.train_params['GCN_Model']['out_channels']

    node_feature_file = config.node_features
    edge_file = config.train_data

    dataset = ACTORNETWORKData_v2(node_feature_file, edge_file)
    in_channels = dataset.x.shape[1]

    model = GCN_Model(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels
    )
    model_path = config.gcn_model_path

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    if link_pairs_to_predict is None:
        if file_path == None:
            file_path = config.test_data
        test_df = pd.read_csv(file_path, sep=' ', header=None, names=['src', 'dst'])
        test_edge_pairs = torch.tensor(test_df[['src', 'dst']].values.T, dtype=torch.long)

    all_preds = []
    with torch.no_grad():
        z = model.encode(dataset.x, dataset.edge_index)
        test_pred = model.decode_mlp(z, test_edge_pairs)
        test_pred = test_pred.cpu().numpy()

    test_pred = (test_pred > 0.5).astype(int)

    return test_pred


def predict_seal_model(model_path=None, file_path=None, link_pairs_to_predict=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_channels = config.train_params['seal_params']['hidden_channels']
    out_channels = config.train_params['seal_params']['out_channels']
    dropout = config.train_params['seal_params']['dropout']

    dataset = get_actors_network_graph()
    in_channels = dataset.x.shape[1]
    node_id_map = dataset.node_id_map

    if link_pairs_to_predict is None:
        if file_path == None:
            file_path = config.test_data
        link_pairs_to_predict = load_test_links(file_path, node_id_map)

    model = SEALModel(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=dropout
    ).to(device)

    if model_path == None or model_path == "seal_model":
        model_path = config.seal_model_path

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    if not isinstance(link_pairs_to_predict, torch.Tensor):
        link_pairs_to_predict = torch.tensor(link_pairs_to_predict, dtype=torch.long)

    link_pairs_to_predict = link_pairs_to_predict.to(device)

    if link_pairs_to_predict.shape[0] == 2:
        link_pairs_to_predict = link_pairs_to_predict.t()

    predict_dataset = [(lp, torch.tensor([0.0])) for lp in link_pairs_to_predict]
    predict_loader = DataLoader(predict_dataset, batch_size=32, shuffle=False)

    all_preds = []

    with torch.no_grad():
        for batch_links, _ in predict_loader:
            batch_links = batch_links.t().to(device)
            preds = model(dataset.x.to(device), dataset.edge_index.to(device), None, batch_links)
            all_preds.append(preds)

    all_preds = torch.cat(all_preds, dim=0)
    predictions = (all_preds > 0.5).float().to("cpu").numpy()
    predictions = list(int(pred) for pred in predictions)

    return predictions


if __name__ == "__main__":
    config = CONFIG()
    # test_path = "data/test.txt"
    test_file_path = config.test_data
    with open(test_file_path, "r") as f:
        reader = csv.reader(f)
        test_set = list(reader)
    test_set = [element[0].split(" ") for element in test_set]
    test_set = [[int(element[0]), int(element[1])] for element in test_set]

    # model_path = config.seal_model_path
    model_path = config.gcn_model_path

    # predictions = predict_seal_model(
    predictions = predict_gcn_model(
        model_path=model_path,
        file_path=test_file_path,
    )
    print("Binary Predictions:", predictions)
    predictions = zip(np.array(range(len(test_set))), predictions)

    data_dir = config.data_dir
    test_predictions_csv = os.path.join(data_dir, "test_predictions_gcn_1000_epoch.csv")
    with open(test_predictions_csv, "w") as pred:
        csv_out = csv.writer(pred)
        csv_out.writerow(i for i in ["ID", "Predicted"])
        for row in predictions:
            csv_out.writerow(row)
        pred.close()
