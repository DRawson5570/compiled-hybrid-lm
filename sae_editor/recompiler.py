from __future__ import annotations

import torch
import torch.nn.functional as F


def build_dense_map(
    keys: torch.Tensor,
    values: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytical matrix construction from key-value pairs.

    Given N key vectors K = [k_1, ..., k_N]^T in R^{N x d_in} and
    N value vectors V = [v_1, ..., v_N]^T in R^{N x d_out}, constructs
    FFN projection matrices such that k_i @ W_down @ W_up = v_i.

    W_down = K^T @ (K @ K^T + eps*I)^{-1}   in R^{d_in x N}
    W_up   = V                               in R^{N x d_out}

    All computation in float32 for numerical stability.

    Args:
        keys:   (N, d_in)  key vectors
        values: (N, d_out) value vectors
        eps:    Tikhonov regularization for matrix inversion

    Returns:
        (W_down, W_up) tuple of tensors in float32
    """
    keys = keys.to(dtype=torch.float32)
    values = values.to(dtype=torch.float32)

    N, d_in = keys.shape
    N_v, d_out = values.shape
    if N != N_v:
        raise ValueError(
            f"Mismatch: {N} keys vs {N_v} values. "
            "Each key must have exactly one value."
        )

    K = keys
    V = values

    gram = K @ K.T
    eye = torch.eye(N, device=gram.device, dtype=torch.float32)
    gram_reg = gram + eps * eye

    L = torch.linalg.cholesky(gram_reg)
    gram_inv = torch.cholesky_inverse(L)

    W_down = K.T @ gram_inv
    W_up = V

    return W_down, W_up


def verify_dense_map(
    keys: torch.Tensor,
    W_down: torch.Tensor,
    W_up: torch.Tensor,
) -> torch.Tensor:
    """Verify that keys @ W_down @ W_up recovers values.

    All computation in float32. Returns the reconstructed values
    as a tensor of shape (N, d_out).
    """
    keys = keys.to(dtype=torch.float32)
    W_down = W_down.to(dtype=torch.float32)
    W_up = W_up.to(dtype=torch.float32)

    return keys @ W_down @ W_up


def orthogonal_projection(
    W_compiled: torch.Tensor,
    U: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Project compiled weights into the orthogonal complement of U.

    Protects existing features from crosstalk by projecting new weights
    into the subspace orthogonal to the original active features.

    P_perp = I - U @ (U^T @ U + eps*I)^{-1} @ U^T
    W_final = P_perp @ W_compiled

    All computation in float32.

    Args:
        W_compiled: (d, k)  compiled weight matrix (e.g. W_down or W_up)
        U:          (d, m)  original active feature directions
        eps:        Tikhonov regularization

    Returns:
        W_final: (d, k) projected weight matrix in float32
    """
    W_compiled = W_compiled.to(dtype=torch.float32)
    U = U.to(dtype=torch.float32)

    d = U.shape[0]
    m = U.shape[1]

    U_cov = U.T @ U
    eye = torch.eye(m, device=U_cov.device, dtype=torch.float32)
    U_cov_reg = U_cov + eps * eye

    L = torch.linalg.cholesky(U_cov_reg)
    U_cov_inv = torch.cholesky_inverse(L)

    P_perp = torch.eye(d, device=U.device, dtype=torch.float32) - U @ U_cov_inv @ U.T

    return P_perp @ W_compiled


def compute_null_space_rank(U: torch.Tensor, eps: float = 1e-6) -> int:
    """Return the rank of the orthogonal complement of U.

    Null space rank = d - rank(U). This is how many independent
    dimensions remain for new patches.
    """
    U = U.to(dtype=torch.float32)
    U_cov = U.T @ U
    eye = torch.eye(U.shape[1], device=U_cov.device, dtype=torch.float32)
    U_cov_reg = U_cov + eps * eye

    L = torch.linalg.cholesky(U_cov_reg)
    U_cov_inv = torch.cholesky_inverse(L)

    P_perp = torch.eye(U.shape[0], device=U.device, dtype=torch.float32) - U @ U_cov_inv @ U.T
    rank = torch.linalg.matrix_rank(P_perp).item()

    return rank


class RecompilerEngine:
    """Phase III: Symbolic-to-Continuous (S2C) recompiler.

    Translates key-value pairs and a feature space specification into
    compiled FFN matrices with crosstalk prevention.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def compile(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        original_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compile key-value pairs into (W_down, W_up) with optional crosstalk prevention.

        Args:
            keys:              (N, d_in)  key vectors
            values:            (N, d_out) value vectors
            original_features: (d_in, m)  active feature directions to protect,
                               or None to skip orthogonal projection

        Returns:
            dict with keys "W_down" and "W_up"
        """
        W_down_raw, W_up_raw = build_dense_map(keys, values, eps=self.eps)

        if original_features is not None and original_features.shape[1] > 0:
            W_down_final = orthogonal_projection(W_down_raw, original_features, eps=self.eps)
        else:
            W_down_final = W_down_raw

        return {
            "W_down": W_down_final,
            "W_up": W_up_raw,
        }

    def compile_from_pairs(
        self,
        pairs: list[tuple[torch.Tensor, torch.Tensor]],
        original_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Convenience: compile from list of (key_vec, value_vec) pairs."""
        keys = torch.stack([k for k, v in pairs], dim=0)
        values = torch.stack([v for k, v in pairs], dim=0)
        return self.compile(keys, values, original_features)

    def verify(self, keys: torch.Tensor, W_down: torch.Tensor, W_up: torch.Tensor) -> torch.Tensor:
        return verify_dense_map(keys, W_down, W_up)


def compact_features(
    W_down: torch.Tensor,
    n_components: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PCA compaction of W_down when null space is low.

    Args:
        W_down:        (d, N) compiled down-projection matrix
        n_components:  Target number of components

    Returns:
        (W_compacted, basis) where W_compacted is (d, n_components)
        and basis transforms W_compacted back to approximate W_down.
    """
    W_down = W_down.to(dtype=torch.float32)

    U, S, Vt = torch.linalg.svd(W_down, full_matrices=False)
    S = S[:n_components]
    Vt = Vt[:n_components, :]
    U = U[:, :n_components]

    basis = torch.diag(S) @ Vt

    W_compacted = U

    return W_compacted, basis


def decompact_features(
    W_compacted: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct W_down from compacted representation."""
    return W_compacted.to(dtype=torch.float32) @ basis.to(dtype=torch.float32)


def pre_activation_scale(
    keys: torch.Tensor,
    activation: str = "gelu",
    target_range: float = 2.0,
) -> torch.Tensor:
    """Scale key vectors to stay in the linear region of activation functions.

    For GELU: linear region is approximately [-2, 2].
    Scales keys such that max(|key_i|) <= target_range.

    Args:
        keys:         (N, d_in) key vectors
        activation:   "gelu", "relu", "silu"
        target_range: Target max absolute value

    Returns:
        Scaled key vectors same shape as input
    """
    keys = keys.to(dtype=torch.float32)
    max_abs = keys.abs().max(dim=-1, keepdim=True).values
    scale = torch.where(
        max_abs > target_range,
        target_range / max_abs,
        torch.ones_like(max_abs),
    )
    return keys * scale
