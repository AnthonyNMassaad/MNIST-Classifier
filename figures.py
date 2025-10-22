import matplotlib.pyplot as plt

def show_batch_images(images, labels, n=10):
    fig = plt.figure(figsize=(12, 8))
    for i in range(n):
        ax = fig.add_subplot(2, 5, i+1)
        img = images[i].detach().cpu()
        if img.shape[0] == 1:
            ax.imshow(img.squeeze(0), cmap='gray')
        else:
            ax.imshow(img.permute(1, 2, 0))
        ax.set_title(f'Label: {labels[i].item()}')
        ax.axis('off')
    plt.show()

def visualize_linear_weights(lin_model):
    import torch
    with torch.no_grad():
        W = lin_model.fc.weight.detach().cpu().numpy()
    fig, axes = plt.subplots(2,5, figsize=(10,5))
    for c, ax in enumerate(axes.ravel()):
        ax.imshow(W[c].reshape(28,28), cmap='seismic')
        ax.set_title(f"W class {c}")
        ax.axis('off')
    plt.suptitle("Linear Classifier Weights")
    plt.tight_layout()
    plt.show()
