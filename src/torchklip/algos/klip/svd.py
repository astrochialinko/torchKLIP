# File: src/torchklip/algos/klip/svd.py
import torch


def compute_basis_svd(reference: torch.Tensor, K_max: int) -> torch.Tensor:
    """Compute KL basis using SVD."""
    _, _, vt = torch.linalg.svd(reference, full_matrices=False)
    return vt.T[:, :K_max]
