import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {"accuracy": acc, "precision": prec, "f1": f1}


def confusion_matrix_np(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(
    cm,
    class_names=None,
    normalize=False,
    title="Confusion Matrix",
    outfile=None,
    show=False,
):
    cm_to_show = cm.astype(np.float64)
    if normalize:
        row_sums = cm_to_show.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_to_show = cm_to_show / row_sums

    C = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(C)]

    fig = plt.figure(figsize=(6.5, 5.5))
    im = plt.imshow(cm_to_show, interpolation="nearest")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(np.arange(C), class_names, rotation=45, ha="right")
    plt.yticks(np.arange(C), class_names)

    thresh = cm_to_show.max() / 2.0 if cm_to_show.size else 0
    for i in range(C):
        for j in range(C):
            txt = f"{cm_to_show[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            plt.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if cm_to_show[i, j] > thresh else "black",
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
