import numpy as np
from config import N_CLASSES

def knn(x_train, y_train, x_test_batch, k):
    train_norm = np.sum(x_train**2, axis=1)
    test_norm  = np.sum(x_test_batch**2, axis=1)
    d = test_norm[:, None] + train_norm[None, :] - 2.0 * (x_test_batch @ x_train.T)
    np.maximum(d, 0, out=d)
    nn_idx = np.argpartition(d, kth=k-1, axis=1)[:, :k]
    nn_labels = y_train[nn_idx]
    preds = np.empty(nn_labels.shape[0], dtype=np.int64)
    for i, row in enumerate(nn_labels):
        counts = np.bincount(row, minlength=N_CLASSES)
        preds[i] = np.argmax(counts)
    return preds

def run_knn(x_train, y_train, x_test, k=3, batch_size=512):
    preds_list = []
    N = x_test.shape[0]
    for s in range(0, N, batch_size):
        e = min(N, s+batch_size)
        preds_list.append(knn(x_train, y_train, x_test[s:e], k=k))
    return np.concatenate(preds_list)
