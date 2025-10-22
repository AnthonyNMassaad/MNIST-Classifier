import matplotlib.pyplot as plt
import torch


def show_batch_images(images, labels, n=10, outfile=None, show=False):
    fig = plt.figure(figsize=(12, 8))
    n = min(n, 10)
    rows = 2
    cols = 5
    for i in range(n):
        ax = fig.add_subplot(rows, cols, i + 1)
        img = images[i].detach().cpu()
        if img.shape[0] == 1:
            ax.imshow(img.squeeze(0), cmap="gray")
        else:
            ax.imshow(img.permute(1, 2, 0))
        ax.set_title(f"Label: {int(labels[i])}")
        ax.axis("off")
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def visualize_linear_weights(lin_model, outfile=None, show=False):
    with torch.no_grad():
        W = lin_model.fc.weight.detach().cpu().numpy()
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for c, ax in enumerate(axes.ravel()):
        ax.imshow(W[c].reshape(28, 28), cmap="seismic")
        ax.set_title(f"W class {c}")
        ax.axis("off")
    plt.suptitle("Linear Classifier Weights")
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_summary_bar(
    results_df,
    metrics=("accuracy", "precision", "f1"),
    title="Model Comparison on MNIST",
    outfile=None,
    show=False,
):
    ax = results_df[list(metrics)].plot(kind="bar", figsize=(11, 6), rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    plt.legend(title="Metric")
    plt.tight_layout()
    if outfile is not None:
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
