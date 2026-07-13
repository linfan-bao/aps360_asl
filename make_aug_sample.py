# make one before/after augmentation figure for the report (figures/aug_sample.png)
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

ROOT = Path(__file__).resolve().parent

def find_data_dir():
    for c in [ROOT / "data/asl_alphabet_train/asl_alphabet_train",
              ROOT / "data/asl_alphabet_train",
              ROOT / "data"]:
        if (c / "A").is_dir():
            return c
    raise FileNotFoundError("ASL train data not found under data/")

# same augmentation as training, minus ToTensor/Normalize so we can actually look at it
aug = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomRotation(15),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

data_dir = find_data_dir()
img = Image.open(data_dir / "A" / "A1.jpg").convert("RGB")

fig, axes = plt.subplots(1, 5, figsize=(14, 3))
axes[0].imshow(img.resize((224, 224))); axes[0].set_title("original"); axes[0].axis("off")
for i in range(1, 5):
    axes[i].imshow(aug(img)); axes[i].set_title(f"augmented {i}"); axes[i].axis("off")
plt.tight_layout()
out = ROOT / "figures" / "aug_sample.png"
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=150)
print(f"saved: {out}")
