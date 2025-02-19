import os
import torch
import random
import numpy as np


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed(seed)  # For CUDA-based computations
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False


BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

class CONFIG:
    def __init__(self):
        self.base_path = BASE_PATH

        self.data_dir = os.path.join(self.base_path, "data")

        # self.raw_data_dir = os.path.join(self.data_dir, "raw")
        # self.processed_data_dir = os.path.join(self.data_dir, "processed")

        self.node_features = os.path.join(self.data_dir, "node_information.csv")
        self.train_data = os.path.join(self.data_dir, "train.txt")
        self.test_data = os.path.join(self.data_dir, "test.txt")


        self.model_dir = os.path.join(BASE_PATH, "models")
        self.sprs_v_cls_model_path = os.path.join(BASE_PATH, "models", "sparse_vector_classifier.pth")

        self.log_dir = os.path.join(BASE_PATH, "logs")

        self.model_name = 'bert-base-uncased'


        self.train_params = {
            'train_size': 0.8,
            'batch_size': 4,
            'epochs': 20,
            'steps_per_epoch': None,
            'latest_checkpoint_step': 50,
            'summary_step': 10,
            'max_checkpoints_to_keep': 5,
            "learning_rate": 0.001,
            "gradient_accumulation_steps": 1,
            'hidden_dim': 16,
        }
        self.model_params = {
            'dropout': 0.2,
        }