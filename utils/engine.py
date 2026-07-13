# reusable train / eval / plot helpers, shared by baseline and primary model
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# same as the ones used in tut and lab
def get_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    model.train()
    return correct / total


def train(model, train_loader, val_loader, device, epochs=5, lr=0.01):
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    crit = nn.CrossEntropyLoss() # multi class classification
    hist = {"epoch": [], "loss": [], "train_acc": [], "val_acc": []}
    for ep in range(epochs):
        model.train()
        running_loss = correct = total = 0
        # implement the training cycle: forward, loss, backward, step (zero grad)
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = crit(out, labels)
            loss.backward()
            opt.step()
            running_loss += loss.item() * labels.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total
        val_acc = get_accuracy(model, val_loader, device)
        hist["epoch"].append(ep + 1)
        hist["loss"].append(running_loss / total)
        hist["train_acc"].append(train_acc)
        hist["val_acc"].append(val_acc)
        print(f"epoch {ep+1}/{epochs}  loss: {running_loss/total:.3f}  train acc: {train_acc:.3f}  val acc: {val_acc:.3f}")
    return hist


def plot_curves(hist, out_path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["epoch"], hist["loss"]); ax[0].set_title("training loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss")
    ax[1].plot(hist["epoch"], hist["train_acc"], label="train")
    ax[1].plot(hist["epoch"], hist["val_acc"], label="val")
    ax[1].set_title("accuracy"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"saved: {out_path}")


def plot_confusion(model, loader, classes, device, out_path):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(device)).argmax(dim=1).cpu()
            y_true += labels.tolist(); y_pred += preds.tolist()
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation="vertical")
    plt.title("confusion matrix"); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"saved: {out_path}")
