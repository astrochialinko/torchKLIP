# File: src/torchklip/algos/klip/eigh.py
import torch


def compute_basis_eigh(reference: torch.Tensor, K_max: int) -> torch.Tensor:
    """Compute KL basis using eigen‐decomposition."""
    cov = reference @ reference.T
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals, eigvecs = eigvals.flip(0), eigvecs.flip(1)
    scales = eigvals[:K_max].sqrt().reciprocal().unsqueeze(0)
    return (reference.T @ (eigvecs[:, :K_max] * scales))
