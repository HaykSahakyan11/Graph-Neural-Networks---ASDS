import torch
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

from src.datasets import get_actors_network_graph
from src.preprocessing import load_link_labels
from src.model import SEALModel, GCNGATModel
from src.config import CONFIG
from src.predict import predict_seal_model

config = CONFIG()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_model(model_name='best_model'):
    in_channels = config.train_params['input_channels']
    if model_name == 'SEAL' or model_name == 'best_model':
        hid_channels = config.train_params['seal_params']['hidden_channels']
        out_channels = config.train_params['seal_params']['out_channels']
        dropout = config.train_params['seal_params']['dropout']
        model = SEALModel(in_channels, hid_channels, out_channels, dropout).to(device)
    elif model_name == 'GCN':
        hid_channels = config.train_params['gcn_params']['hidden_channels']
        out_channels = config.train_params['gcn_params']['out_channels']
        dropout = config.train_params['gcn_params']['dropout']
        model = GCNGATModel(in_channels, hid_channels, out_channels, dropout).to(device)
    return model


def load_model(model_name):
    if model_name == 'best_model':
        model_path = config.best_model_path
    elif model_name == 'SEAL':
        model_path = config.seal_model_path
    elif model_name == 'GCN':
        model_path = config.gcn_gat_model_path

    model = init_model(model_name)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def evaluate(model_name='best_model', eval_file_path=None):
    if eval_file_path is None:
        eval_file_path = config.train_data

    model = load_model(model_name)
    model.eval()

    node_id_map = get_actors_network_graph().node_id_map

    link_pairs, labels = load_link_labels(filepath=eval_file_path, node_id_map=node_id_map)
    labels = labels.to(device)

    predictions = predict_seal_model(
        link_pairs_to_predict=link_pairs
    )

    # Converting predictions to a tensor
    predictions = torch.tensor(predictions, dtype=torch.float, device=device)

    # Calculating metrics
    accuracy = accuracy_score(labels.cpu(), predictions.cpu())
    auc = roc_auc_score(labels.cpu(), predictions.cpu())
    precision, recall, f1, _ = precision_recall_fscore_support(labels.cpu(), predictions.cpu(), average='binary')

    print(f"Evaluation Results: Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    # Saving results to CSV
    results_df = pd.DataFrame([{
        'model_name': model_name,
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }])

    results_path = config.log_dir + f"/{model_name}_evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Evaluation results saved at {results_path}")


if __name__ == '__main__':
    evaluate()
