# data pipeline: load ASL, subsample 800 per class, split train/val, build loaders
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent  # project root (utils/ sits one level down)

# only A-Z, skip space/del/nothing
CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]

# knobs we might tweak later
PER_CLASS = 800       # how many images to keep per class
VAL_RATIO = 0.15      # 85/15 train/val
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# train transform: crop/rotate/blur/jitter then normalize (no flip, it would change the sign)
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomRotation(15),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# val transform: clean, just resize + normalize
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def find_data_dir():
    for c in [ROOT / "data/asl_alphabet_train/asl_alphabet_train",
              ROOT / "data/asl_alphabet_train",
              ROOT / "data"]:
        if (c / "A").is_dir():
            return c
    raise FileNotFoundError("ASL train data not found under data/")


class ASLDataset(torch.utils.data.Dataset):
    def __init__(self, items, transform):
        self.items = items        # list of (path, label)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_items(data_dir):
    # subsample per class, then split per class so train/val stay balanced
    rng = random.Random(SEED)
    train_items, val_items = [], []
    for label, cls in enumerate(CLASSES):
        files = sorted((data_dir / cls).glob("*.jpg"))
        rng.shuffle(files)
        files = files[:PER_CLASS]
        n_val = int(len(files) * VAL_RATIO)
        val_items += [(f, label) for f in files[:n_val]]
        train_items += [(f, label) for f in files[n_val:]]
    return train_items, val_items


def get_loaders():
    data_dir = find_data_dir()
    train_items, val_items = build_items(data_dir)
    train_loader = DataLoader(ASLDataset(train_items, train_tf),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(ASLDataset(val_items, val_tf),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader


if __name__ == "__main__":
    tl, vl = get_loaders()
    print(f"train images: {len(tl.dataset)}  val images: {len(vl.dataset)}")
    print(f"train batches: {len(tl)}  val batches: {len(vl)}")
    xb, yb = next(iter(tl))
    print(f"one batch: {tuple(xb.shape)}  labels: {yb[:8].tolist()}")
