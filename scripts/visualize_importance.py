"""
Visualize token importance scores for one image.

Usage:
    python scripts/visualize_importance.py \
        --checkpoint ./checkpoints/xxx/split/edge_weights.pth \
        --image ./data/coco/val/000000000139.jpg \
        --out_dir ./checkpoints/xxx/importance_vis
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.edge_client import EdgeEncoder


def _save_overlay(image_np, value_map, out_path, title, cmap="jet", alpha=0.45, vmin=None, vmax=None):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.imshow(image_np)
    ax.imshow(value_map, cmap=cmap, alpha=alpha, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize importance map from edge checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to edge checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--out_dir", type=str, default="./checkpoints/importance_vis", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = EdgeEncoder(checkpoint_path=args.checkpoint, device=args.device)
    if not encoder.importance_aware:
        raise ValueError("Checkpoint does not enable importance_aware.")

    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)
    image_tensor = encoder.transform(image).unsqueeze(0).to(encoder.device)

    with torch.no_grad():
        feat = encoder.student(image_tensor)[-1]
        tokens = encoder.projector(feat)
        logits = encoder.importance_scorer(tokens)
        probs = torch.sigmoid(logits)

        k = max(
            encoder.budgeted_transmission.min_tokens,
            min(encoder.budgeted_transmission.target_budget, encoder.budgeted_transmission.max_tokens),
        )
        k = min(k, tokens.size(1))
        _, topk_idx = logits.topk(k, dim=1, sorted=True)

    n_tokens = tokens.size(1)
    side = int(math.sqrt(n_tokens))
    if side * side != n_tokens:
        raise ValueError(f"Token count {n_tokens} is not a square grid.")

    probs_np = probs.squeeze(0).detach().cpu().numpy().reshape(side, side)
    logits_np = logits.squeeze(0).detach().cpu().numpy().reshape(side, side)
    mask = np.zeros(n_tokens, dtype=np.float32)
    mask[topk_idx.squeeze(0).detach().cpu().numpy()] = 1.0
    mask = mask.reshape(side, side)

    h, w = image_np.shape[:2]
    probs_up = np.array(Image.fromarray((probs_np * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)) / 255.0
    mask_up = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST)) / 255.0

    stem = Path(args.image).stem
    _save_overlay(
        image_np=image_np,
        value_map=probs_up,
        out_path=out_dir / f"{stem}_importance_prob_overlay.png",
        title=f"Importance probability overlay (k={k}/{n_tokens})",
        cmap="jet",
        alpha=0.45,
        vmin=0.0,
        vmax=1.0,
    )
    _save_overlay(
        image_np=image_np,
        value_map=mask_up,
        out_path=out_dir / f"{stem}_topk_mask_overlay.png",
        title=f"Top-K mask overlay (k={k}/{n_tokens})",
        cmap="Reds",
        alpha=0.45,
        vmin=0.0,
        vmax=1.0,
    )

    np.save(out_dir / f"{stem}_importance_probs_grid.npy", probs_np)
    np.save(out_dir / f"{stem}_importance_logits_grid.npy", logits_np)
    np.save(out_dir / f"{stem}_topk_mask_grid.npy", mask)

    print(f"Saved: {out_dir / f'{stem}_importance_prob_overlay.png'}")
    print(f"Saved: {out_dir / f'{stem}_topk_mask_overlay.png'}")
    print(f"Saved grids (.npy) in: {out_dir}")
    print(
        f"Stats: tokens={n_tokens}, k={k}, prob_min={probs_np.min():.4f}, "
        f"prob_mean={probs_np.mean():.4f}, prob_max={probs_np.max():.4f}"
    )


if __name__ == "__main__":
    main()
