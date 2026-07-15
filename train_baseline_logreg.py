# baseline: a single linear layer over flattened pixels (logistic regression)
# no convolution, no hidden layer -- our simplest reference model
from pathlib import Path
import torch
import torch.nn as nn

from utils.data import get_loaders, CLASSES, IMG_SIZE
from utils.engine import train, plot_curves, plot_confusion

ROOT = Path(__file__).resolve().parent
TAG = "baseline_logreg"   # bump this each experiment so figures/weights aren't overwritten


class LinearBaseline(nn.Module):
    # flatten the image and classify with one linear layer (no conv, no hidden units)
    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        in_features = 3 * IMG_SIZE * IMG_SIZE   # 3 x 224 x 224 = 150528
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)   # [B, 3*224*224]
        return self.fc(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_loader, val_loader = get_loaders()
    model = LinearBaseline()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params: {total:,}  trainable: {trainable:,}")

    hist = train(model, train_loader, val_loader, device, epochs=15, lr=0.01)
    print("final val accuracy:", hist["val_acc"][-1])

    (ROOT / "models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), ROOT / "models" / f"{TAG}.pth")
    plot_curves(hist, ROOT / "figures" / f"{TAG}_curve.png")
    plot_confusion(model, val_loader, CLASSES, device, ROOT / "figures" / f"{TAG}_confusion.png")
    print("done")
