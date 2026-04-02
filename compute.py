"""
Loss Landscape Computation
Implements the filter-normalized direction method from:
  "Visualizing the Loss Landscape of Neural Nets" (Li et al., 2018)
  https://arxiv.org/abs/1712.09913

Key insight: naive random directions in weight space are misleading because
different layers have vastly different weight scales. Filter normalization
rescales each direction vector to match the norm of the corresponding filter,
making landscapes comparable across architectures and training runs.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from copy import deepcopy


# ── Direction generation ──────────────────────────────────────────────────────

def get_random_direction(model: nn.Module) -> List[torch.Tensor]:
    """
    Sample a random direction in weight space with the same structure as
    the model's parameters. Each direction vector is a list of tensors,
    one per parameter.
    """
    direction = []
    for param in model.parameters():
        d = torch.randn_like(param)
        direction.append(d)
    return direction


def filter_normalize(direction: List[torch.Tensor], model: nn.Module) -> List[torch.Tensor]:
    """
    Filter normalization (Li et al. 2018, Section 4.1).

    For each parameter tensor, normalize each filter (or row for 2D params)
    to have the same norm as the corresponding filter in the model weights.

    This ensures the perturbation size is meaningful relative to the actual
    weight magnitudes — without this, directions are dominated by layers with
    large weights, making the landscape uninterpretable.
    """
    normalized = []
    for d, param in zip(direction, model.parameters()):
        if param.dim() == 1:
            # Bias or batch norm: scalar normalization
            param_norm = param.norm().item()
            d_norm = d.norm().item()
            scale = param_norm / (d_norm + 1e-10)
            normalized.append(d * scale)

        elif param.dim() == 2:
            # Linear weights: normalize each row (output neuron)
            d_scaled = torch.zeros_like(d)
            for i in range(d.shape[0]):
                param_norm = param[i].norm().item()
                d_norm = d[i].norm().item()
                scale = param_norm / (d_norm + 1e-10)
                d_scaled[i] = d[i] * scale
            normalized.append(d_scaled)

        elif param.dim() >= 3:
            # Conv weights: normalize each filter (dim 0)
            d_scaled = torch.zeros_like(d)
            for i in range(d.shape[0]):
                param_norm = param[i].norm().item()
                d_norm = d[i].norm().item()
                scale = param_norm / (d_norm + 1e-10)
                d_scaled[i] = d[i] * scale
            normalized.append(d_scaled)

        else:
            normalized.append(d)

    return normalized


def get_normalized_directions(model: nn.Module) -> Tuple[List, List]:
    """Return two orthogonal, filter-normalized random directions."""
    d1 = filter_normalize(get_random_direction(model), model)
    d2 = filter_normalize(get_random_direction(model), model)
    return d1, d2


# ── Weight space perturbation ─────────────────────────────────────────────────

def set_weights(model: nn.Module, base_weights: List[torch.Tensor],
                d1: List[torch.Tensor], d2: List[torch.Tensor],
                alpha: float, beta: float):
    """
    Set model weights to: theta* + alpha * d1 + beta * d2
    where theta* are the trained weights (the point we're exploring around).
    """
    with torch.no_grad():
        for param, w, v1, v2 in zip(model.parameters(), base_weights, d1, d2):
            param.copy_(w + alpha * v1 + beta * v2)


def restore_weights(model: nn.Module, base_weights: List[torch.Tensor]):
    """Restore model to its trained weights."""
    with torch.no_grad():
        for param, w in zip(model.parameters(), base_weights):
            param.copy_(w)


# ── Landscape computation ─────────────────────────────────────────────────────

def compute_landscape(
    model: nn.Module,
    d1: List[torch.Tensor],
    d2: List[torch.Tensor],
    dataloader,
    loss_fn,
    device: torch.device,
    grid_size: int = 30,
    range_: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate loss on a 2D grid of weight perturbations.

    For each (alpha, beta) in [-range_, range_]^2:
        theta = theta* + alpha * d1 + beta * d2
        loss  = evaluate(model_with_theta, data)

    Returns: alphas [G], betas [G], losses [G, G]

    grid_size=30 means 30×30 = 900 loss evaluations.
    With a small eval dataset this takes ~2-5 minutes on CPU.
    """
    base_weights = [p.data.clone() for p in model.parameters()]
    model.eval()

    alphas = np.linspace(-range_, range_, grid_size)
    betas  = np.linspace(-range_, range_, grid_size)
    losses = np.zeros((grid_size, grid_size))

    total = grid_size * grid_size
    done = 0

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            set_weights(model, base_weights, d1, d2, alpha, beta)

            batch_loss = 0.0
            batch_count = 0
            with torch.no_grad():
                for x, y in dataloader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    loss = loss_fn(out, y)
                    batch_loss += loss.item() * x.size(0)
                    batch_count += x.size(0)

            losses[i, j] = batch_loss / batch_count
            done += 1
            if done % 50 == 0:
                print(f"  Grid: {done}/{total}  loss@(0,0)={losses[grid_size//2, grid_size//2]:.4f}", end='\r')

    restore_weights(model, base_weights)
    print(f"  Grid complete. Loss range: [{losses.min():.4f}, {losses.max():.4f}]")
    return alphas, betas, losses


def compute_training_trajectory(
    model: nn.Module,
    d1: List[torch.Tensor],
    d2: List[torch.Tensor],
    checkpoints: List[List[torch.Tensor]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project training checkpoints onto the 2D landscape directions.
    Lets us draw the optimization path on top of the loss surface.

    Returns: alphas [T], betas [T] for each checkpoint
    """
    final_weights = [p.data.clone() for p in model.parameters()]
    traj_alphas, traj_betas = [], []

    for ckpt_weights in checkpoints:
        # delta = checkpoint - final_weights
        delta = [c - w for c, w in zip(ckpt_weights, final_weights)]

        # Project delta onto d1, d2
        # alpha = <delta, d1> / <d1, d1>
        dot_d1 = sum((dt * v).sum().item() for dt, v in zip(delta, d1))
        dot_d2 = sum((dt * v).sum().item() for dt, v in zip(delta, d2))
        norm_d1 = sum((v * v).sum().item() for v in d1)
        norm_d2 = sum((v * v).sum().item() for v in d2)

        traj_alphas.append(dot_d1 / (norm_d1 + 1e-10))
        traj_betas.append(dot_d2 / (norm_d2 + 1e-10))

    return np.array(traj_alphas), np.array(traj_betas)
