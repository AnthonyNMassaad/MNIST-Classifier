import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from config import device

def eval_accuracy(model, loader):
    model.eval()
    correct = total = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total   += yb.numel()
            y_true.append(yb.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
    return correct / total, y_true, y_pred

def train_cnn(model, train_ds, test_ds, epochs=15, lr=0.1, batch_train=128, batch_test=512):
    train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_test,  shuffle=False, num_workers=0)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss   = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)

        tr_acc, _, _ = eval_accuracy(model, train_loader)
        te_acc, _, _ = eval_accuracy(model, test_loader)
        print(f"Epoch {epoch} - loss {running/len(train_ds):.4f} - train_acc {tr_acc:.4f} - test_acc {te_acc:.4f}")

    # predictions
    model.eval()
    all_preds = []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
    y_pred_cnn = np.concatenate(all_preds)
    return y_pred_cnn
