# File: src/torchklip/algos/klip/pca.py
import torch


def compute_basis_pca(reference: torch.Tensor, K_max: int) -> torch.Tensor:
    """Compute KL basis using PCA (low‐rank)."""
    _, _, V = torch.pca_lowrank(reference, q=K_max, center=False)
    return V
