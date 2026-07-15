# baseline: a small fully-connected network (MLP) over flattened pixels
# one hidden layer, no convolution -- a simple reference model
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data import get_loaders, CLASSES, IMG_SIZE
from utils.engine import train, plot_curves, plot_confusion

ROOT = Path(__file__).resolve().parent
TAG = "baseline_mlp"   # bump this each experiment so figures/weights aren't overwritten


class SmallMLP(nn.Module):
    # flatten the image, one hidden layer, then classify (no conv layers)
    def __init__(self, hidden=256, num_classes=len(CLASSES)):
        super().__init__()
        in_features = 3 * IMG_SIZE * IMG_SIZE   # 3 x 224 x 224 = 150528
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)        # [B, 3*224*224]
        x = F.relu(self.fc1(x))
        return self.fc2(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_loader, val_loader = get_loaders()
    model = SmallMLP()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params: {total:,}  trainable: {trainable:,}")

    # small lr: a linear-heavy model on raw pixels needs a gentle step to stay stable
    hist = train(model, train_loader, val_loader, device, epochs=20, lr=1e-4)
    print("final val accuracy:", hist["val_acc"][-1])

    (ROOT / "models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), ROOT / "models" / f"{TAG}.pth")
    plot_curves(hist, ROOT / "figures" / f"{TAG}_curve.png")
    plot_confusion(model, val_loader, CLASSES, device, ROOT / "figures" / f"{TAG}_confusion.png")
    print("done")
