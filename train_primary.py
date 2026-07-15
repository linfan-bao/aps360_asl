#primary model: ResNet-18 based ASL classifier
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from utils.data import (get_loaders, CLASSES, ASLDataset, val_tf,
                        build_items, find_data_dir)
from utils.engine import train, get_accuracy, plot_curves, plot_confusion

ROOT = Path(__file__).resolve().parent
TAG = "primary_ep20"   # bump this each experiment so figures/weights aren't overwritten


class ASLNet(nn.Module):
    def __init__(self, hidden=256, dropout=0.2, num_classes=len(CLASSES)):
        super().__init__()
        #use resnet-18 (conv + global avg pool) as a frozen feature extractor
        backbone = resnet18(weights=None) #construct a n empty resnet18 model
        backbone.load_state_dict(torch.load(ROOT / "models" / "resnet18-f37072fd.pth")) # load in pretrained weights which allows transfer learning
        backbone.fc = nn.Identity() # discard the originial classifier and let it output the features classified
        self.backbone = backbone
        for p in self.backbone.parameters(): # freeze the backbone
            p.requires_grad = False
        
        # asl classifier head (small one)
        self.fc1 = nn.Linear(512, hidden)
        self.drop = nn.Dropout(dropout) # p=0.5
        self.fc2 = nn.Linear(hidden, num_classes)
        #give the new layers initial weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        self.backbone.eval() # switch frozen backbone to eval mode
        with torch.no_grad():
            features = self.backbone(x)        # conv + global avg pool -> [B, 512]
        h = F.relu(self.fc1(features))
        h = self.drop(h)
        return self.fc2(h)


def sanity_check(device):
    # can our head overfit 64 mixed samples? train acc should climb high
    items = build_items(find_data_dir())[0]
    random.Random(0).shuffle(items)
    loader = DataLoader(ASLDataset(items[:64], val_tf), batch_size=32, shuffle=True)
    model = ASLNet().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    for _ in range(40):
        model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = crit(model(imgs), labels)
            loss.backward()
            opt.step()
    print(f"sanity check: overfit 64 -> train acc {get_accuracy(model, loader, device):.3f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    print("--- sanity check ---")
    sanity_check(device)

    print("--- full training ---")
    train_loader, val_loader = get_loaders()
    model = ASLNet()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total params: {total:,}  trainable: {trainable:,}")

    hist = train(model, train_loader, val_loader, device, epochs=20, lr=0.01)
    print("final val accuracy:", hist["val_acc"][-1])

    torch.save(model.state_dict(), ROOT / "models" / f"{TAG}.pth")
    plot_curves(hist, ROOT / "figures" / f"{TAG}_curve.png")
    plot_confusion(model, val_loader, CLASSES, device, ROOT / "figures" / f"{TAG}_confusion.png")
    print("done")
