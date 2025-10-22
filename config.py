import torch
import numpy as np
import random
import os

ROOT_DIR = "MNIST"
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
TEST_SIZE = 0.2
IMG_SIZE = (28, 28)
RANDOM_SEED = 42
TARGET_SIZE = None
N_CLASSES = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
