import numpy as np
from scipy.linalg import cho_factor, cho_solve

def make_nystrom_precond_apply(U: np.ndarray, lam_hat: np.ndarray, mu: float):
    """
    Returns function apply_Pinv(r) = P^{-1} r
    from paper eq (5.3):
      P^{-1} = (λ̂_ell + μ) U (Λ̂ + μI)^{-1} U^T + (I - U U^T)
    """
    lam_ell = float(lam_hat[-1])
    scale = lam_ell + mu
    denom = lam_hat + mu

    def apply_Pinv(r: np.ndarray) -> np.ndarray:
        Ut_r = U.T @ r
        partU = U @ (scale * (Ut_r / denom))
        proj = U @ Ut_r
        return partU + (r - proj)

    return apply_Pinv

import scipy.linalg

def make_block_nystrom_precond_apply(C_blocks: list, W_blocks: list, lambda_reg: float):
    """
    CORRECT: Block-Nyström preconditioner using Woodbury identity.

    For Block-Nyström: Ĵ_block = C * (q * block_diag(W_i))^{-1} * C^T
    We need: P^{-1} = (Ĵ_block + λI)^{-1}

    Using Woodbury: (λI + C * (q*W_block)^{-1} * C^T)^{-1} =
        = λ^{-1}[I - C(q*W_block + λ^{-1} C^T C)^{-1} C^T λ^{-1}]
    """
    q = len(C_blocks)
    n = C_blocks[0].shape[0]

    # 1. Stack C matrices: C = [C₁, C₂, ..., C_q]
    C = np.hstack(C_blocks)  # (n, m_total) where m_total = sum(b_i)
    m_total = C.shape[1]

    # 2. Build q*W_block where W_block = block_diag(W₁, W₂, ..., W_q)
    W_diag_blocks = []
    for W_i in W_blocks:
        b_i = W_i.shape[0]
        # Regularize each block
        W_i_reg = W_i + 1e-8 * np.eye(b_i)
        # Multiply by q as in Block-Nyström formula
        W_diag_blocks.append(q * W_i_reg)

    # Create block diagonal matrix
    W_block = scipy.linalg.block_diag(*W_diag_blocks)  # (m_total, m_total)

    # 3. Precompute matrix for Woodbury formula:
    # M = (q*W_block + λ^{-1} C^T C)
    CtC = C.T @ C  # (m_total, m_total)
    M = W_block + (1.0 / lambda_reg) * CtC

    # Add regularization and invert
    M_reg = M + 1e-8 * np.eye(M.shape[0])
    M_inv = np.linalg.inv(M_reg)

    # 4. Woodbury formula:
    # P^{-1}r = λ^{-1}[r - C * M^{-1} * C^T * (λ^{-1} r)]

    def apply_Pinv(r: np.ndarray) -> np.ndarray:
        """
        Compute P^{-1} * r where P = Ĵ_block + λI
        """
        lambda_inv_r = r / lambda_reg
        Ct_lambda_inv_r = C.T @ lambda_inv_r  # C^T * (λ^{-1} r)
        correction_term = M_inv @ Ct_lambda_inv_r  # M^{-1} * C^T * (λ^{-1} r)
        C_correction = C @ correction_term  # C * M^{-1} * C^T * (λ^{-1} r)

        return (r / lambda_reg) - C_correction

    return apply_Pinv

def rp_cholesky_factor(A: np.ndarray, ell: int, seed: int = 0, candidate_pool: int = 50, eps: float = 1e-12):
    """
    Randomized pivoted Cholesky for PSD matrix A (dense).

    Returns:
      L: (n, r) with r<=ell, pivots (list), diag_res (residual diagonal)
    """
    n = A.shape[0]
    rng = np.random.default_rng(seed)

    diag_res = np.clip(np.diag(A).copy(), 0.0, None)
    L = np.zeros((n, ell), dtype=A.dtype)
    pivots = []

    for j in range(ell):
        s = float(diag_res.sum())
        if (not np.isfinite(s)) or (s <= eps):
            L = L[:, :j]
            break

        pool = min(candidate_pool, n)
        probs = diag_res / s
        cand = rng.choice(n, size=pool, replace=False, p=probs)
        p = int(cand[np.argmax(diag_res[cand])])

        delta = float(diag_res[p])
        if delta <= eps:
            L = L[:, :j]
            break

        pivots.append(p)

        if j == 0:
            w = A[:, p].copy()
        else:
            w = A[:, p] - L[:, :j] @ L[p, :j]

        L[:, j] = w / np.sqrt(delta)
        diag_res = np.clip(diag_res - L[:, j]**2, 0.0, None)

    return L, pivots, diag_res

def make_rpchol_precond_apply(L: np.ndarray, lam: float):
    """
    Preconditioner apply_Pinv(r) for (lam I + L L^T)^{-1} via Woodbury:
      (lam I + L L^T)^{-1} r = (1/lam) r - (1/lam^2) L (I + (1/lam) L^T L)^{-1} L^T r
    """
    if L.size == 0:
        return lambda r: (1.0 / lam) * r

    G = np.eye(L.shape[1]) + (1.0 / lam) * (L.T @ L)
    c, low = cho_factor(G, check_finite=False)

    def apply_Pinv(r: np.ndarray) -> np.ndarray:
        t = L.T @ r
        z = cho_solve((c, low), t, check_finite=False)
        return (1.0 / lam) * r - (1.0 / (lam * lam)) * (L @ z)

    return apply_Pinv
