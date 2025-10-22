import numpy as np
from config import N_CLASSES

def binarize(X, threshold=0.5):
    return (X > threshold).astype(np.uint8)

def fit_bernoulli(Xb, y, n_classes=N_CLASSES, alpha=1.0):
    N, D = Xb.shape
    class_counts = np.bincount(y, minlength=n_classes)
    priors = (class_counts + 1) / (N + n_classes)

    theta = np.empty((n_classes, D), dtype=np.float64)
    for c in range(n_classes):
        Xc = Xb[y == c]
        theta[c] = (Xc.sum(axis=0) + alpha) / (Xc.shape[0] + 2.0 * alpha)

    eps = 1e-12
    theta = np.clip(theta, eps, 1 - eps)
    return priors.astype(np.float64), theta

def predict_bernoulli(Xb, priors, theta, batch_size=2048):
    log_priors = np.log(priors)
    log_theta = np.log(theta)
    log_1m_theta = np.log(1.0 - theta)
    N = Xb.shape[0]
    preds = np.empty(N, dtype=np.int64)
    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        Xbatch = Xb[s:e]
        ll = (Xbatch @ log_theta.T) + ((1 - Xbatch) @ log_1m_theta.T)
        log_post = ll + log_priors[None, :]
        preds[s:e] = np.argmax(log_post, axis=1)
    return preds
