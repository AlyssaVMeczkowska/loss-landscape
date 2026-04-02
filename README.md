# Loss Landscape Visualization

A from-scratch implementation of the filter-normalized loss landscape method from Li et al. (2018), applied to ResNet-20 on CIFAR-10. Visualizes why optimizer choice and architecture matter for generalization.

<img width="1383" height="1184" alt="demo_3d" src="https://github.com/user-attachments/assets/eb5f6aff-fa90-4fcb-a5c5-1dfc667d057c" />
<img width="1133" height="1035" alt="demo_2d" src="https://github.com/user-attachments/assets/46d24b5d-5a0d-4aed-acc8-6a9acaff8e6d" />


The loss surface of a neural network lives in a space with millions of dimensions — impossible to visualize directly. Li et al.'s key insight: pick two random directions in weight space, perturb the trained weights along those directions, and plot the resulting loss. With filter normalization, the directions are scaled to match the actual weight magnitudes, making landscapes comparable.

The result reveals a striking empirical fact:

> SGD with momentum finds flatter minima than Adam. Flatter minima generalize better — small perturbations to the weights don't change the loss much, which corresponds to robustness to distribution shift.

## Methodology

Naive random directions are dominated by layers with large weights. Filter normalization fixes this by rescaling each direction vector filter-by-filter:

$$\hat{d}_i^{(l)} = \frac{d_i^{(l)}}{\|d_i^{(l)}\|} \cdot \|\theta_i^{(l)}\|$$

where $i$ indexes filters (rows of weight matrices, output channels of conv layers). This ensures the perturbation is proportional to the weight magnitude at every layer.

The landscape is then computed as:

$$\mathcal{L}(\alpha, \beta) = \mathcal{L}(\theta^* + \alpha \hat{d}_1 + \beta \hat{d}_2)$$

evaluated on a grid $(\alpha, \beta) \in [-r, r]^2$.

## Architecture

| File | Description |
|---|---|
| `landscape/compute.py` | Filter normalization, direction sampling, grid evaluation, trajectory projection |
| `landscape/models.py` | SmallResNet (skip connections) and VGGStyle (no skip) for comparison |
| `landscape/train.py` | Training loop with checkpoint saving for trajectory visualization |
| `landscape/visualize.py` | 3D surface, 2D contour + trajectory, SGD vs Adam comparison, sharpness metric |
| `main.py` | CLI entry point |

## Quick start

```bash
pip install -r requirements.txt

# quick demo 
python main.py --mode demo

# full experiment on CIFAR-10 
python main.py --mode all --epochs 20 --grid 25

# separately
python main.py --mode train --epochs 20
python main.py --mode visualize --grid 30
```

## Figures generated

| Figure | Description |
|---|---|
| `landscape_3d_sgd.png` | 3D loss surface around SGD minimum |
| `landscape_3d_adam.png` | 3D loss surface around Adam minimum |
| `landscape_2d_sgd.png` | 2D contour + training trajectory (SGD) |
| `landscape_2d_adam.png` | 2D contour + training trajectory (Adam) |
| `comparison.png` | Side-by-side SGD vs Adam + sharpness bar chart |
| `training_curves.png` | Loss and accuracy curves for both optimizers |

## Key findings

- SGD (with cosine LR and momentum) finds flatter minima — the loss surface is smooth and bowl-shaped
- Adam tends to find sharper minima — narrow valleys with steep walls
- Skip connections (ResNet) produce significantly flatter landscapes than plain networks (VGG-style)
- The training trajectory projected onto the 2D landscape shows how SGD spirals slowly into a flat basin while Adam converges quickly to a sharp point

## References

- Li et al. (2018) — [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913)
- Keskar et al. (2017) — [On Large-Batch Training: Sharpness and Generalization](https://arxiv.org/abs/1609.04836)
- Izmailov et al. (2018) — [Averaging Weights Leads to Wider Optima (SWA)](https://arxiv.org/abs/1803.05407)
