# baseline: a small CNN trained from scratch (no pretraining), lab3 style
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data import get_loaders, CLASSES
from utils.engine import train, plot_curves, plot_confusion

ROOT = Path(__file__).resolve().parent
TAG = "baseline_cnn_ep20"   # bump this each experiment so figures/weights aren't overwritten


class SmallCNN(nn.Module):
    # simple conv stack, everything trained from random init (our baseline)
    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        # 4 conv blocks: 224 -> 112 -> 56 -> 28 -> 14
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # classifier head: 128 x 14 x 14 -> 256 -> 26
        self.fc1 = nn.Linear(128 * 14 * 14, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # -> [B,16,112,112]
        x = self.pool(F.relu(self.conv2(x)))   # -> [B,32,56,56]
        x = self.pool(F.relu(self.conv3(x)))   # -> [B,64,28,28]
        x = self.pool(F.relu(self.conv4(x)))   # -> [B,128,14,14]
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_loader, val_loader = get_loaders()
    model = SmallCNN()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params: {total:,}  trainable: {trainable:,}")

    # from scratch needs more epochs than the frozen-backbone models to converge
    hist = train(model, train_loader, val_loader, device, epochs=20, lr=0.01)
    print("final val accuracy:", hist["val_acc"][-1])

    (ROOT / "models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), ROOT / "models" / f"{TAG}.pth")
    plot_curves(hist, ROOT / "figures" / f"{TAG}_curve.png")
    plot_confusion(model, val_loader, CLASSES, device, ROOT / "figures" / f"{TAG}_confusion.png")
    print("done")
