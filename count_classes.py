# doing a summary of the number of images per class in the ASL Alphabet dataset
from pathlib import Path
import matplotlib.pyplot as plt

# resolve everything from this script's folder, not the current working dir
ROOT = Path(__file__).resolve().parent

# we only care about A-Z, not space or del
CLASSES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]

def find_data_dir():
    """Automatically locate the directory containing A/ B/ ... subdirectories (Kaggle extraction results in nested directories with the same names)"""
    for c in [ROOT / "data/asl_alphabet_train/asl_alphabet_train",
              ROOT / "data/asl_alphabet_train",
              ROOT / "data"]:
        if (c / "A").is_dir():
            return c
    return None

def count_images(data_dir):
    exts = {".jpg", ".jpeg", ".png"}
    counts = {}
    for cls in CLASSES:
        d = data_dir / cls
        counts[cls] = sum(1 for f in d.iterdir() if f.suffix.lower() in exts) if d.is_dir() else 0
    return counts

def main():
    data_dir = find_data_dir()
    if data_dir is None:
        print("Data directory not found.")
        return
    print(f"Data directory: {data_dir.resolve()}")
    counts = count_images(data_dir)
    total = sum(counts.values())
    print(f"Number of classes: {sum(1 for n in counts.values() if n>0)}  Total images: {total}")
    for cls, n in counts.items():
        print(f"  {cls}: {n}")

    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.bar(list(counts.keys()), list(counts.values()))
    plt.title("ASL Alphabet - Images per Class (train)")
    plt.xlabel("Class"); plt.ylabel("Number of images")
    plt.tight_layout()
    plt.savefig(figures_dir / "class_counts.png", dpi=150)
    print("✓ Bar chart saved: figures/class_counts.png")

if __name__ == "__main__":
    main()
