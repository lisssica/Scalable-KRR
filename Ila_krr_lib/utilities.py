import numpy as np
from scipy.sparse.linalg import eigsh

def effective_dimension(K: np.ndarray, mu: float, *, max_exact_n: int = 4000, top_m: int = 600, eps: float = 1e-12):
    """
    Estimate d_eff(mu) = tr(K (K + mu I)^{-1}) = sum_i lambda_i/(lambda_i+mu).

    Returns dict with:
      d_eff_est : float  (top-part estimate)
      d_eff_lb  : float  (lower bound, equals top-part)
      d_eff_ub  : float  (upper bound using trace remainder / mu)
      traceK    : float
      used_m    : int
    """
    n = K.shape[0]
    mu = float(mu)
    trK = float(np.trace(K))

    if n <= max_exact_n:
        w = np.linalg.eigvalsh(0.5*(K + K.T))
        w = np.clip(w, 0.0, None)
        d_eff = float(np.sum(w / (w + mu)))
        return {"d_eff_est": d_eff, "d_eff_lb": d_eff, "d_eff_ub": d_eff, "traceK": trK, "used_m": n}

    m = min(top_m, n - 2)
    w = eigsh(K, k=m, which="LM", return_eigenvectors=False)
    w = np.sort(w)[::-1]
    w = np.clip(w, 0.0, None)

    d_top = float(np.sum(w / (w + mu)))
    sum_top = float(np.sum(w))
    rem = max(trK - sum_top, 0.0)

    d_ub = d_top + rem / max(mu, eps)
    return {"d_eff_est": d_top, "d_eff_lb": d_top, "d_eff_ub": float(d_ub), "traceK": trK, "used_m": m}

def suggest_ell_from_deff(d_eff: float) -> int:
    """
    Paper-style heuristic:
      ell = 2 * ceil(1.5 * d_eff) + 1
    """
    return int(2 * np.ceil(1.5 * float(d_eff)) + 1)

def median_heuristic_gamma(X: np.ndarray, n_pairs: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    d2 = np.sum((X[i] - X[j])**2, axis=1)
    med = float(np.median(d2))
    return 1.0 / (2.0 * med + 1e-30)

def median_length_scale(X: np.ndarray, n_pairs: int = 20000, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    d2 = np.sum((X[i] - X[j])**2, axis=1)
    med = float(np.median(d2))
    return float(np.sqrt(med) + 1e-12)
