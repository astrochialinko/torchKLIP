# File: src/torchklip/algos/klip/eigh.py
import torch


def compute_basis_eigh(reference: torch.Tensor, K_max: int) -> torch.Tensor:
    """Compute KL basis using eigen-decomposition of the Gram matrix.

    Args:
        reference: ``(N, P)`` library matrix (N frames, P pixels).
        K_max: Number of KL modes to return.  Must satisfy ``K_max <= N``.

    Returns:
        ``(P, K_max)`` matrix (same dtype as ``reference``) whose columns are
        the orthonormal KL modes.

    Raises:
        ValueError: If ``K_max`` exceeds the number of library frames.

    Note:
        Forming ``reference @ reference.T`` squares the condition number of the
        problem.  In float32 the eigenvectors of modes whose eigenvalue is below
        roughly ``1e-7 * lambda_0`` are essentially noise, which matters for
        high-dynamic-range data (e.g. a bright PSF core): results for large K
        can then differ by tens of percent from a float64 computation.  The
        Gram matrix, its eigendecomposition and the basis are therefore computed
        in float64 and cast back to the input dtype.  MPS has no float64 support,
        so on MPS the computation stays in float32 -- prefer ``method="svd"``
        there for high-dynamic-range data.

        Eigenvalues are additionally floored at a relative tolerance
        (``eps * N * lambda_0``) so that ``K_max`` close to the library rank
        cannot produce inf/NaN.
    """
    n = reference.shape[0]
    if K_max > n:
        raise ValueError(
            f"K_max={K_max} exceeds the number of library frames ({n}).")

    work = reference if reference.device.type == "mps" else reference.to(torch.float64)

    cov = work @ work.T
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals, eigvecs = eigvals.flip(0), eigvecs.flip(1)

    finfo = torch.finfo(eigvals.dtype)
    floor = max(float(eigvals[0]) * finfo.eps * n, finfo.tiny)
    scales = (
        torch.clamp(eigvals[:K_max], min=floor).sqrt().reciprocal().unsqueeze(0)
    )
    basis = work.T @ (eigvecs[:, :K_max] * scales)
    return basis.to(reference.dtype)