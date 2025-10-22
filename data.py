import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from config import ROOT_DIR, TRAIN_BATCH_SIZE, TEST_SIZE, RANDOM_SEED

# Your transforms and loaders preserved
mnist_tf = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ]
)


def get_loader(root=ROOT_DIR, batch_size=64, flatten=False):
    tfms = [mnist_tf]
    if flatten:
        tfms.append(transforms.Lambda(lambda t: t.view(-1)))
    ds = datasets.ImageFolder(root=root, transform=transforms.Compose(tfms))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)


def load_preview_batch():
    cnn_loader = get_loader(ROOT_DIR, batch_size=TRAIN_BATCH_SIZE, flatten=False)
    flat_loader = get_loader(ROOT_DIR, batch_size=TRAIN_BATCH_SIZE, flatten=True)
    x_cnn, y_cnn = next(iter(cnn_loader))
    x_flat, y_flat = next(iter(flat_loader))
    return (cnn_loader, x_cnn, y_cnn), (flat_loader, x_flat, y_flat)


def make_numpy_partitions(flat_loader, test_size=TEST_SIZE, seed=RANDOM_SEED):
    x_list, y_list = [], []
    for x, y in flat_loader:
        x_list.append(x.numpy())
        y_list.append(y.numpy())
    x_all = np.concatenate(x_list, axis=0).astype(np.float32)
    y_all = np.concatenate(y_list, axis=0).astype(np.int64)

    s = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(s.split(np.zeros(len(y_all)), y_all))

    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_test, y_test = x_all[test_idx], y_all[test_idx]
    return x_all, y_all, x_train, y_train, x_test, y_test


def to_tensor(x, y):
    xt = torch.from_numpy(x.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.int64))
    return xt, yt


def to_image_tensors(x_train, y_train, x_test, y_test):
    xtr_img = torch.from_numpy(x_train.reshape(-1, 1, 28, 28).astype(np.float32))
    ytr = torch.from_numpy(y_train.astype(np.int64))
    xte_img = torch.from_numpy(x_test.reshape(-1, 1, 28, 28).astype(np.float32))
    yte = torch.from_numpy(y_test.astype(np.int64))
    train_ds = TensorDataset(xtr_img, ytr)
    test_ds = TensorDataset(xte_img, yte)
    return train_ds, test_ds
