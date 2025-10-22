import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import device, N_CLASSES, TRAIN_BATCH_SIZE, set_seed
from data import (
    get_loader,
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
from figures import show_batch_images, visualize_linear_weights


def main():
    print("\nEnvironment")
    print("Using device:", device)
    set_seed()

    print("\nData Loading and Preview")
    (cnn_loader, x_cnn, y_cnn), (flat_loader, x_flat, y_flat) = load_preview_batch()
    print("CNN batch shape:", x_cnn.shape)
    print("Flat batch shape:", x_flat.shape)
    print("CNN loader pixel range:", float(x_cnn.min()), "to", float(x_cnn.max()))
    print("Flat loader pixel range:", float(x_flat.min()), "to", float(x_flat.max()))

    print("\nCreating Train/Test Partitions")
    # Split into numpy train/test
    x_all, y_all, x_train, y_train, x_test, y_test = make_numpy_partitions(flat_loader)
    print("Train shape:", x_train.shape)
    print("Test shape: ", x_test.shape)

    # KNN
    print("\nK-Nearest Neighbors (KNN)")
    print("Running k-NN for multiple k values...")
    k_values = [1, 3, 5]
    knn_metrics = {}
    for k in k_values:
        print(f"  - k = {k}: predicting...")
        y_pred_knn_k = run_knn(x_train, y_train, x_test, k=k)
        metrics_knn_k = compute_metrics(y_test, y_pred_knn_k)
        knn_metrics[f"KNN(k={k})"] = metrics_knn_k
        print(f"    metrics: {metrics_knn_k}")

    # Naive Bayes
    print("\nNaive Bayes (Bernoulli)")
    print("Binarizing data with threshold 0.5 and fitting Bernoulli NB...")
    x_train_bin = binarize(x_train, 0.5)
    x_test_bin = binarize(x_test, 0.5)
    priors, theta = fit_bernoulli(x_train_bin, y_train, n_classes=N_CLASSES, alpha=1.0)
    y_pred_nb = predict_bernoulli(x_test_bin, priors, theta)
    metrics_nb = compute_metrics(y_test, y_pred_nb)
    print("Naive Bayes metrics:", metrics_nb)

    cm_nb = confusion_matrix_np(y_test, y_pred_nb, n_classes=N_CLASSES)

    # Linear
    print("\nLinear Classifier (L2 Loss)")
    print("Preparing tensors and training linear model...")
    xtr_lin, ytr_lin = to_tensor(x_train, y_train)
    xte_lin, yte_lin = to_tensor(x_test, y_test)
    lin_model = LinearClassifier(in_dim=x_train.shape[1])
    y_pred_lin = train_linear(
        lin_model, xtr_lin, ytr_lin, xte_lin, yte_lin, epochs=20, batch=128, lr=0.1
    )
    metrics_lin = compute_metrics(y_test, y_pred_lin)
    print("Linear model metrics:", metrics_lin)
    # visualize_linear_weights(lin_model)

    # MLP
    print("\nMLP")
    print("Initializing MLP and training...")
    mlp_model = MLP(in_dim=x_train.shape[1], h1=256, h2=128)
    y_pred_mlp = train_classifier(
        mlp_model, xtr_lin, ytr_lin, xte_lin, yte_lin, epochs=20, batch=128, lr=0.1
    )
    metrics_mlp = compute_metrics(y_test, y_pred_mlp)
    print("MLP metrics:", metrics_mlp)

    # CNN
    print("\nConvolutional Neural Network (CNN)")
    print("Converting to image tensors and training CNN...")
    train_ds_cnn, test_ds_cnn = to_image_tensors(x_train, y_train, x_test, y_test)
    cnn_model = SimpleCNN()
    y_pred_cnn = train_cnn(cnn_model, train_ds_cnn, test_ds_cnn, epochs=15, lr=0.1)
    metrics_cnn = compute_metrics(y_test, y_pred_cnn)
    print("CNN metrics:", metrics_cnn)

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
    print("Aggregated results:\n", results)

    metrics_to_plot = ["accuracy", "precision", "f1"]
    ax = results[metrics_to_plot].plot(kind="bar", figsize=(11, 6), rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on MNIST")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
