import os
import torch
import random
import numpy as np
import sklearn.utils


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    sklearn.utils.check_random_state(seed)  # scikit-learn
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed(seed)  # For CUDA-based computations
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False


BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class CONFIG:
    def __init__(self):
        self.base_path = BASE_PATH

        self.data_dir = os.path.join(self.base_path, "data")

        self.node_features = os.path.join(self.data_dir, "node_information.csv")
        self.train_data = os.path.join(self.data_dir, "train.txt")
        self.test_data = os.path.join(self.data_dir, "test.txt")

        self.model_dir = os.path.join(BASE_PATH, "models")
        self.best_model_path = os.path.join(self.model_dir, "SEAL_epoch_79_best_acc_0.803.pth")
        # self.best_model_path = os.path.join(self.model_dir, "SEAL_epoch_80_best_acc_0.829.pth")
        self.gcn_gat_model_path = os.path.join(self.model_dir, "GCN_GAT_epoch_76_best_acc_0.686.pth")
        self.seal_model_path = os.path.join(self.model_dir, "SEAL_epoch_79_best_acc_0.803.pth")

        self.log_dir = os.path.join(BASE_PATH, "logs")

        self.model_name = 'bert-base-uncased'

        self.train_params = {
            'train_size': 0.80,
            'batch_size': 4,
            'epochs': 20,
            "learning_rate": 0.001,
            'hidden_dim': 16,
            'input_channels': 932,
            'gcn_params': {
                'hidden_channels': 256,
                'out_channels': 2,
                'dropout': 0.2,
                'threshold': 0.5,
                'lr': 0.01,
                'weight_decay': 5e-4,
                'epochs': 100,
            },
            'seal_params': {
                'hidden_channels': 256,
                'out_channels': 64,
                'dropout': 0.3,
                'threshold': 0.5,
                'lr': 0.0001,
                'weight_decay': 5e-4,
                'epochs': 100,
                # 'epochs': 5,
            },
        }
