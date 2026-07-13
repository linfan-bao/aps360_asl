# draw the primary model block diagram for the report -> figures/primary_diagram.png
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent

blocks = [
    ("input image\nRGB 224x224", "#eaeaea"),
    ("ResNet-18 backbone\n(frozen) -> 512-d", "#cfe3f7"),
    ("our classifier head\nFC 512->256, ReLU,\nDropout 0.2, FC 256->26", "#fde0c2"),
    ("prediction\nargmax -> A-Z", "#eaeaea"),
]

fig, ax = plt.subplots(figsize=(13, 3.4))
ax.set_xlim(0, 44); ax.set_ylim(0, 10); ax.axis("off")

w, h, gap = 9, 4.2, 2
x, edges = 1, []
for text, color in blocks:
    ax.add_patch(FancyBboxPatch((x, 3.3), w, h, boxstyle="round,pad=0.1",
                                fc=color, ec="#555555", lw=1.2))
    ax.text(x + w / 2, 5.4, text, ha="center", va="center", fontsize=10)
    edges.append((x, x + w)); x += w + gap

for i in range(len(blocks) - 1):
    ax.annotate("", xy=(edges[i + 1][0], 5.4), xytext=(edges[i][1], 5.4),
                arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#555555"))

ax.add_patch(Rectangle((1, 1), 0.9, 0.9, fc="#cfe3f7", ec="#555555"))
ax.text(2.2, 1.45, "frozen (pretrained on ImageNet)", va="center", fontsize=9)
ax.add_patch(Rectangle((22, 1), 0.9, 0.9, fc="#fde0c2", ec="#555555"))
ax.text(23.2, 1.45, "trainable (our design, Xavier init)", va="center", fontsize=9)

ax.set_title("primary model: ResNet-18 features + our classifier head", fontsize=12)
plt.tight_layout()
out = ROOT / "figures" / "primary_diagram.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved: {out}")
