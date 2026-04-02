"""
Loss Landscape Visualization
Generates publication-quality figures:
1. 3D surface plot of the loss landscape
2. 2D contour with optimization trajectory overlay
3. Side-by-side SGD vs Adam landscape comparison
4. Training curves
5. Sharpness metric comparison
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec

from landscape.models import SmallResNet, VGGStyle
from landscape.compute import (
    get_normalized_directions, compute_landscape, compute_training_trajectory
)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
BG     = "#080b0f"
PANEL  = "#0e1318"
ACCENT = "#00d9ff"
GREEN  = "#39d98a"
AMBER  = "#ffb547"
PINK   = "#ff6b9d"

CMAP_LANDSCAPE = mcolors.LinearSegmentedColormap.from_list(
    "landscape", ["#0a1628", "#1a3a6e", "#2e6eb5", "#4ab3e8", "#7de8ff", "#ffffff"]
)
CMAP_SHARP = mcolors.LinearSegmentedColormap.from_list(
    "sharp", ["#0a1628", "#3a1040", "#8b1a6e", "#e03080", "#ff9050", "#ffffff"]
)


def style_ax3d(ax):
    ax.set_facecolor(PANEL)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#1e2d3d")
    ax.yaxis.pane.set_edgecolor("#1e2d3d")
    ax.zaxis.pane.set_edgecolor("#1e2d3d")
    ax.tick_params(colors="#3a5068", labelsize=7)
    ax.xaxis.label.set_color("#3a5068")
    ax.yaxis.label.set_color("#3a5068")
    ax.zaxis.label.set_color("#3a5068")


def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors="#3a5068", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e2d3d")


def fig_3d_surface(alphas, betas, losses, title="Loss Landscape", save_path="figures/landscape_3d.png"):
    """3D surface plot — the money shot."""
    A, B = np.meshgrid(alphas, betas, indexing="ij")
    L = np.log1p(losses - losses.min() + 0.01)  # log scale for visual clarity

    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor(BG)
    ax = fig.add_subplot(111, projection="3d")
    style_ax3d(ax)

    surf = ax.plot_surface(A, B, L, cmap=CMAP_LANDSCAPE, alpha=0.92,
                           linewidth=0, antialiased=True, rcount=60, ccount=60)

    # Mark the minimum
    min_idx = np.unravel_index(np.argmin(losses), losses.shape)
    ax.scatter([alphas[min_idx[0]]], [betas[min_idx[1]]], [L[min_idx]],
               color=GREEN, s=80, zorder=10, label="Minimum")

    ax.set_xlabel("Direction 1 (α)", labelpad=8)
    ax.set_ylabel("Direction 2 (β)", labelpad=8)
    ax.set_zlabel("log(Loss)", labelpad=8)
    ax.set_title(title, color="white", fontsize=13, pad=15)
    ax.view_init(elev=35, azim=-60)

    cb = fig.colorbar(surf, ax=ax, shrink=0.4, aspect=12, pad=0.1)
    cb.ax.tick_params(colors="#3a5068", labelsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {save_path}")


def fig_2d_contour(alphas, betas, losses, traj_a=None, traj_b=None,
                   title="Loss Contour", save_path="figures/landscape_2d.png"):
    """2D filled contour with optional optimization trajectory."""
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(BG)
    style_ax(ax)

    A, B = np.meshgrid(alphas, betas, indexing="ij")
    L = np.log1p(losses - losses.min() + 0.01)

    cf = ax.contourf(A, B, L, levels=40, cmap=CMAP_LANDSCAPE)
    ax.contour(A, B, L, levels=15, colors="white", alpha=0.12, linewidths=0.5)

    # Trajectory
    if traj_a is not None and traj_b is not None and len(traj_a) > 1:
        ax.plot(traj_a, traj_b, color=AMBER, lw=2, alpha=0.9, zorder=5)
        ax.scatter(traj_a[:-1], traj_b[:-1], color=AMBER, s=30, zorder=6, alpha=0.7)
        ax.scatter(traj_a[-1],  traj_b[-1],  color=GREEN,  s=80, zorder=7,
                   marker="*", label="Final weights")
        ax.scatter(traj_a[0],   traj_b[0],   color=PINK,   s=60, zorder=7,
                   marker="o", label="Init")

    # Center marker (theta*)
    ax.scatter([0], [0], color=GREEN, s=120, zorder=8, marker="*",
               label="θ* (trained)" if traj_a is None else None)

    cb = fig.colorbar(cf, ax=ax)
    cb.ax.tick_params(colors="#3a5068", labelsize=7)
    cb.set_label("log(Loss)", color="#3a5068", fontsize=8)

    ax.set_xlabel("Direction 1 (α)", color="#3a5068")
    ax.set_ylabel("Direction 2 (β)", color="#3a5068")
    ax.set_title(title, color="white", fontsize=12)
    if traj_a is not None:
        ax.legend(facecolor=PANEL, edgecolor="#1e2d3d", labelcolor="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {save_path}")


def fig_comparison(results_sgd, results_adam, save_path="figures/comparison.png"):
    """
    Side-by-side comparison of SGD vs Adam loss landscapes.
    The key visual: SGD typically finds flatter minima, Adam sharper ones.
    """
    fig = plt.figure(figsize=(18, 7))
    fig.patch.set_facecolor(BG)
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    axes_2d = [fig.add_subplot(gs[0, i]) for i in range(2)]
    ax_sharp = fig.add_subplot(gs[0, 2])

    configs = [
        (results_sgd,  "SGD (cosine LR)", CMAP_LANDSCAPE),
        (results_adam, "Adam",             CMAP_SHARP),
    ]

    sharpness = {}
    for ax, (res, label, cmap) in zip(axes_2d, configs):
        style_ax(ax)
        alphas, betas, losses = res
        A, B = np.meshgrid(alphas, betas, indexing="ij")
        L = np.log1p(losses - losses.min() + 0.01)

        ax.contourf(A, B, L, levels=35, cmap=cmap)
        ax.contour(A, B, L, levels=12, colors="white", alpha=0.1, linewidths=0.4)
        ax.scatter([0], [0], color=GREEN, s=100, zorder=5, marker="*")
        ax.set_title(label, color="white", fontsize=11)
        ax.set_xlabel("α", color="#3a5068")
        ax.set_ylabel("β", color="#3a5068")

        # Sharpness = max eigenvalue of Hessian proxy = max loss in neighborhood
        center_i = len(alphas) // 2
        neighborhood = losses[center_i-3:center_i+4, center_i-3:center_i+4]
        sharpness[label] = float(neighborhood.max() - neighborhood.min())

    # Sharpness bar chart
    style_ax(ax_sharp)
    names = list(sharpness.keys())
    vals  = list(sharpness.values())
    colors_bar = [ACCENT, PINK]
    bars = ax_sharp.bar(names, vals, color=colors_bar, width=0.5, edgecolor="#1e2d3d")
    for bar, val in zip(bars, vals):
        ax_sharp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=9)
    ax_sharp.set_title("Sharpness (local loss range)", color="white", fontsize=11)
    ax_sharp.set_ylabel("max − min loss in neighborhood", color="#3a5068")
    ax_sharp.set_ylim(0, max(vals) * 1.3)
    ax_sharp.tick_params(colors="#3a5068")

    fig.suptitle("Loss Landscape Comparison: SGD vs Adam", color="white", fontsize=14, y=1.02)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {save_path}")


def fig_training_curves(sgd_data, adam_data, save_path="figures/training_curves.png"):
    """Training loss and test accuracy for both optimizers."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(BG)

    for ax in axes:
        style_ax(ax)
        ax.grid(alpha=0.08)

    sgd_losses, sgd_accs   = sgd_data
    adam_losses, adam_accs = adam_data

    e_sgd  = range(1, len(sgd_losses)  + 1)
    e_adam = range(1, len(adam_losses) + 1)

    axes[0].plot(e_sgd,  sgd_losses,  color=ACCENT, lw=2, label="SGD")
    axes[0].plot(e_adam, adam_losses, color=PINK,   lw=2, label="Adam")
    axes[0].set_title("Training Loss", color="white")
    axes[0].set_xlabel("Epoch", color="#3a5068")
    axes[0].legend(facecolor=PANEL, edgecolor="#1e2d3d", labelcolor="white")

    axes[1].plot(e_sgd,  sgd_accs,  color=ACCENT, lw=2, label="SGD")
    axes[1].plot(e_adam, adam_accs, color=PINK,   lw=2, label="Adam")
    axes[1].set_title("Test Accuracy", color="white")
    axes[1].set_xlabel("Epoch", color="#3a5068")
    axes[1].legend(facecolor=PANEL, edgecolor="#1e2d3d", labelcolor="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {save_path}")


def generate_all(checkpoint_dir="checkpoints", grid_size=25, device=None):
    """Load trained models and generate all figures."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    os.makedirs("figures", exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_ds     = datasets.CIFAR10("./data", train=False, download=True, transform=test_transform)
    eval_subset = Subset(test_ds, range(1000))
    eval_loader = DataLoader(eval_subset, batch_size=128, shuffle=False)
    loss_fn     = nn.CrossEntropyLoss()

    results = {}
    for opt in ["sgd", "adam"]:
        ckpt_path = os.path.join(checkpoint_dir, f"resnet_{opt}.pt")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path} — run train.py first")
            continue

        ckpt = torch.load(ckpt_path, map_location=device)
        model = SmallResNet().to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        print(f"\nComputing {opt.upper()} landscape ({grid_size}×{grid_size} grid)...")
        d1, d2 = get_normalized_directions(model)
        d1 = [v.to(device) for v in d1]
        d2 = [v.to(device) for v in d2]

        alphas, betas, losses = compute_landscape(
            model, d1, d2, eval_loader, loss_fn, device,
            grid_size=grid_size, range_=0.8
        )

        results[opt] = {
            "alphas": alphas, "betas": betas, "losses": losses,
            "train_losses": ckpt["train_losses"],
            "test_accs":    ckpt["test_accs"],
            "checkpoints":  ckpt["checkpoints"],
            "model": model, "d1": d1, "d2": d2,
        }

        fig_3d_surface(alphas, betas, losses,
                       title=f"Loss Landscape — ResNet ({opt.upper()})",
                       save_path=f"figures/landscape_3d_{opt}.png")

        # Trajectory
        if ckpt["checkpoints"]:
            traj_a, traj_b = compute_training_trajectory(
                model, d1, d2,
                [[w.to(device) for w in c] for c in ckpt["checkpoints"]]
            )
        else:
            traj_a = traj_b = None

        fig_2d_contour(alphas, betas, losses, traj_a, traj_b,
                       title=f"Loss Contour + Training Trajectory ({opt.upper()})",
                       save_path=f"figures/landscape_2d_{opt}.png")

    if "sgd" in results and "adam" in results:
        fig_comparison(
            (results["sgd"]["alphas"],  results["sgd"]["betas"],  results["sgd"]["losses"]),
            (results["adam"]["alphas"], results["adam"]["betas"], results["adam"]["losses"]),
        )
        fig_training_curves(
            (results["sgd"]["train_losses"],  results["sgd"]["test_accs"]),
            (results["adam"]["train_losses"], results["adam"]["test_accs"]),
        )

    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    generate_all()
