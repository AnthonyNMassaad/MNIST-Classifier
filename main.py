import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import device, N_CLASSES, TRAIN_BATCH_SIZE, set_seed
from data import (
    load_preview_batch,
    make_numpy_partitions,
    to_tensor,
    to_image_tensors,
)
from metrics import compute_metrics, confusion_matrix_np, plot_confusion_matrix
from knn import run_knn
from naive_bayes import binarize, fit_bernoulli, predict_bernoulli
from models import LinearClassifier, MLP, SimpleCNN
from trainer import train_linear, train_classifier
from cnn_train import train_cnn
from figures import (
    show_batch_images,
    visualize_linear_weights,
    plot_summary_bar,
)


def plot_confusion_matrix_subplot(cm, title="Confusion Matrix"):
    cm_normalized = cm.astype(np.float64)
    row_sums = cm_normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_normalized = cm_normalized / row_sums

    im = plt.imshow(cm_normalized, interpolation="nearest", cmap="Blues")
    plt.title(title, fontsize=10)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    C = cm.shape[0]
    tick_marks = np.arange(C)
    plt.xticks(tick_marks, range(C), fontsize=8)
    plt.yticks(tick_marks, range(C), fontsize=8)

    thresh = cm_normalized.max() / 2.0
    for i in range(C):
        for j in range(C):
            plt.text(
                j,
                i,
                f"{cm_normalized[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if cm_normalized[i, j] > thresh else "black",
            )

    plt.ylabel("True", fontsize=9)
    plt.xlabel("Pred", fontsize=9)


def main():
    print("\nEnvironment")
    print("Using device:", device)
    set_seed()

    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    print("\nData Loading and Preview")
    (cnn_loader, x_cnn, y_cnn), (flat_loader, x_flat, y_flat) = load_preview_batch()
    show_batch_images(
        x_cnn,
        y_cnn,
        n=10,
        outfile=os.path.join(out_dir, "batch_grid.png"),
        show=False,
    )

    print("CNN batch shape:", x_cnn.shape)
    print("Flat batch shape:", x_flat.shape)
    print("CNN loader pixel range:", float(x_cnn.min()), "to", float(x_cnn.max()))
    print("Flat loader pixel range:", float(x_flat.min()), "to", float(x_flat.max()))

    print("\nCreating Train/Test Partitions")
    x_all, y_all, x_train, y_train, x_test, y_test = make_numpy_partitions(flat_loader)
    print("Train shape:", x_train.shape)
    print("Test shape: ", x_test.shape)

    # KNN
    print("\nKNN")
    print("Running KNN...")
    k_values = [1, 3, 5]
    knn_metrics = {}
    knn_predictions = {}
    for k in k_values:
        print(f"  - k = {k}: predicting...")
        y_pred_knn_k = run_knn(x_train, y_train, x_test, k=k)
        knn_predictions[k] = y_pred_knn_k
        metrics_knn_k = compute_metrics(y_test, y_pred_knn_k)
        knn_metrics[f"KNN(k={k})"] = metrics_knn_k
        print(f"    metrics: {metrics_knn_k}")
        cm_knn = confusion_matrix_np(y_test, y_pred_knn_k, n_classes=N_CLASSES)
        plot_confusion_matrix(
            cm_knn,
            class_names=[str(i) for i in range(N_CLASSES)],
            normalize=False,
            title=f"Confusion Matrix - KNN(k={k})",
            outfile=os.path.join(out_dir, f"confusion_knn_k{k}.png"),
            show=False,
        )

    # Naive Bayes
    print("\nNaive Bayes")
    print("Binarizing data with threshold 0.5...")
    x_train_bin = binarize(x_train, 0.5)
    x_test_bin = binarize(x_test, 0.5)
    priors, theta = fit_bernoulli(x_train_bin, y_train, n_classes=N_CLASSES, alpha=1.0)
    y_pred_nb = predict_bernoulli(x_test_bin, priors, theta)
    metrics_nb = compute_metrics(y_test, y_pred_nb)
    print("Naive Bayes metrics:", metrics_nb)

    cm_nb = confusion_matrix_np(y_test, y_pred_nb, n_classes=N_CLASSES)
    plot_confusion_matrix(
        cm_nb,
        class_names=[str(i) for i in range(N_CLASSES)],
        normalize=False,
        title="Confusion Matrix - Naive Bayes",
        outfile=os.path.join(out_dir, "confusion_naive_bayes.png"),
        show=False,
    )

    # Linear
    print("\nLinear Classifier")
    print("Preparing tensors and training linear model...")
    xtr_lin, ytr_lin = to_tensor(x_train, y_train)
    xte_lin, yte_lin = to_tensor(x_test, y_test)
    lin_model = LinearClassifier(in_dim=x_train.shape[1])
    y_pred_lin = train_linear(
        lin_model, xtr_lin, ytr_lin, xte_lin, yte_lin, epochs=20, batch=128, lr=0.1
    )
    metrics_lin = compute_metrics(y_test, y_pred_lin)
    print("Linear model metrics:", metrics_lin)

    # linear model weights
    visualize_linear_weights(
        lin_model,
        outfile=os.path.join(out_dir, "linear_weights.png"),
        show=False,
    )

    cm_lin = confusion_matrix_np(y_test, y_pred_lin, n_classes=N_CLASSES)
    plot_confusion_matrix(
        cm_lin,
        class_names=[str(i) for i in range(N_CLASSES)],
        normalize=False,
        title="Confusion Matrix - Linear",
        outfile=os.path.join(out_dir, "confusion_linear.png"),
        show=False,
    )

    # MLP
    print("\nMLP")
    print("Initializing MLP and training...")
    mlp_model = MLP(in_dim=x_train.shape[1], h1=256, h2=128)
    y_pred_mlp = train_classifier(
        mlp_model, xtr_lin, ytr_lin, xte_lin, yte_lin, epochs=20, batch=128, lr=0.1
    )
    metrics_mlp = compute_metrics(y_test, y_pred_mlp)
    print("MLP metrics:", metrics_mlp)

    cm_mlp = confusion_matrix_np(y_test, y_pred_mlp, n_classes=N_CLASSES)
    plot_confusion_matrix(
        cm_mlp,
        class_names=[str(i) for i in range(N_CLASSES)],
        normalize=False,
        title="Confusion Matrix - MLP",
        outfile=os.path.join(out_dir, "confusion_mlp.png"),
        show=False,
    )

    # CNN
    print("\nCNN")
    print("Converting to image tensors and training CNN...")
    train_ds_cnn, test_ds_cnn = to_image_tensors(x_train, y_train, x_test, y_test)
    cnn_model = SimpleCNN()
    y_pred_cnn = train_cnn(cnn_model, train_ds_cnn, test_ds_cnn, epochs=15, lr=0.1)
    metrics_cnn = compute_metrics(y_test, y_pred_cnn)
    print("CNN metrics:", metrics_cnn)

    cm_cnn = confusion_matrix_np(y_test, y_pred_cnn, n_classes=N_CLASSES)
    plot_confusion_matrix(
        cm_cnn,
        class_names=[str(i) for i in range(N_CLASSES)],
        normalize=False,
        title="Confusion Matrix - CNN",
        outfile=os.path.join(out_dir, "confusion_cnn.png"),
        show=False,
    )

    # Summary
    print("\nSummary of Test Metrics")
    results = pd.DataFrame.from_dict(
        {
            **knn_metrics,
            "NaiveBayes": metrics_nb,
            "Linear": metrics_lin,
            "MLP": metrics_mlp,
            "CNN": metrics_cnn,
        },
        orient="index",
    )

    print(results)

    plot_summary_bar(
        results,
        metrics=("accuracy", "precision", "f1"),
        title="Model Comparison on MNIST",
        outfile=os.path.join(out_dir, "metrics_comparison.png"),
        show=False,
    )

    print(f"\nSaved figures to: {out_dir}")


if __name__ == "__main__":
    main()
