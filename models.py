import torch.nn as nn
from config import N_CLASSES

class LinearClassifier(nn.Module):
    def __init__(self, in_dim=784, n_classes=N_CLASSES):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(1)
        return self.fc(x)

class MLP(nn.Module):
    def __init__(self, in_dim=784, h1=256, h2=128, n_classes=N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, n_classes)
        )
    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(1)
        return self.net(x)

class SimpleCNN(nn.Module):
    def __init__(self, n_classes=N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
