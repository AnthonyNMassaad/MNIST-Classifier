import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from config import device, N_CLASSES

def one_hot(y, n_classes=N_CLASSES):
    oh = torch.zeros((y.shape[0], n_classes), dtype=torch.float32)
    oh[torch.arange(y.shape[0]), y] = 1.0
    return oh

# Linear model with L2 loss (as required)
def train_linear(model, xtr, ytr, xte, yte, epochs=20, batch=128, lr=0.1):
    model = model.to(device)
    criterion = nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=lr)

    ds = TensorDataset(xtr, ytr)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)

    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb_oh = one_hot(yb, N_CLASSES).to(device)
            logits = model(xb)
            loss = criterion(logits, yb_oh)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item() * xb.size(0)

        model.eval()
        with torch.no_grad():
            tr_pred = model(xtr.to(device)).argmax(1).cpu().numpy()
            te_pred = model(xte.to(device)).argmax(1).cpu().numpy()
        tr_acc = accuracy_score(ytr.cpu().numpy(), tr_pred)
        te_acc = accuracy_score(yte.cpu().numpy(), te_pred)
        print(f"Epoch {ep} - loss {running/len(xtr):.4f} - train_acc {tr_acc:.4f} - test_acc {te_acc:.4f}")

    with torch.no_grad():
        y_pred = model(xte.to(device)).argmax(1).cpu().numpy()
    return y_pred

# Generic classifier trainer (CrossEntropy) reused by MLP
def train_classifier(model, xtr, ytr, xte, yte, epochs=15, batch=128, lr=0.1):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    train_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=batch, shuffle=True, num_workers=0)
    for ep in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item() * xb.size(0)
        model.eval()
        with torch.no_grad():
            tr_pred = model(xtr.to(device)).argmax(1).cpu().numpy()
            te_pred = model(xte.to(device)).argmax(1).cpu().numpy()
        tr_acc = accuracy_score(ytr.cpu().numpy(), tr_pred)
        te_acc = accuracy_score(yte.cpu().numpy(), te_pred)
        print(f"Epoch {ep} - loss {running_loss/len(xtr):.4f} - train_acc {tr_acc:.4f} - test_acc {te_acc:.4f}")
    with torch.no_grad():
        y_pred = model(xte.to(device)).argmax(1).cpu().numpy()
    return y_pred
