# baseline: pretrained GoogLeNet, freeze features, train a new 26-class head
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import googlenet

from utils.data import get_loaders, CLASSES
from utils.engine import train, plot_curves, plot_confusion

ROOT = Path(__file__).resolve().parent
TAG = "baseline_ep8"   # bump this each experiment so figures/weights aren't overwritten


def build_model():
    # load pretrained weights from a local file (torch's ssl download fails on this machine)
    model = googlenet(weights=None, init_weights=False)
    model.load_state_dict(torch.load(ROOT / "models" / "googlenet-1378be20.pth"))
    # drop aux classifiers so forward returns a plain tensor
    model.aux_logits = False
    model.aux1 = None
    model.aux2 = None
    model.transform_input = False   # we already normalize with ImageNet stats in data.py
    # freeze the pretrained backbone
    for p in model.parameters():
        p.requires_grad = False
    # new head (classifier for asl): 1024 -> 26
    model.fc = nn.Linear(1024, len(CLASSES))
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_loader, val_loader = get_loaders()
    model = build_model()

    hist = train(model, train_loader, val_loader, device, epochs=8, lr=0.01)
    print("final val accuracy:", hist["val_acc"][-1])

    (ROOT / "models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), ROOT / "models" / f"{TAG}.pth")
    plot_curves(hist, ROOT / "figures" / f"{TAG}_curve.png")
    plot_confusion(model, val_loader, CLASSES, device, ROOT / "figures" / f"{TAG}_confusion.png")
    print("done")
