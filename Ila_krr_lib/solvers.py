import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve

def pcg(A_apply, b, M_apply=None, x0=None, tol=1e-6, max_iter=1000):
    """
    Preconditioned Conjugate Gradient for SPD system A x = b.
    Stopping: ||r||/||b|| <= tol.
    Returns: x, iters, final_rel_res
    """
    n = b.shape[0]
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    r = b - A_apply(x)
    bnorm = np.linalg.norm(b) + 1e-30
    rel = np.linalg.norm(r) / bnorm
    if rel <= tol:
        return x, 0, rel

    if M_apply is None:
        z = r.copy()
    else:
        z = M_apply(r)

    p = z.copy()
    rz_old = float(r @ z)

    for it in range(1, max_iter + 1):
        Ap = A_apply(p)
        alpha = rz_old / (float(p @ Ap) + 1e-30)
        x += alpha * p
        r -= alpha * Ap

        rel = np.linalg.norm(r) / bnorm
        if rel <= tol:
            return x, it, rel

        if M_apply is None:
            z = r.copy()
        else:
            z = M_apply(r)

        rz_new = float(r @ z)
        beta = rz_new / (rz_old + 1e-30)
        p = z + beta * p
        rz_old = rz_new

    return x, max_iter, rel

def krr_exact_solve(Ktr: np.ndarray, ytr: np.ndarray, lam):
    """Solve A alpha = y via Cholesky. Returns alpha and solve_time."""
    n = Ktr.shape[0]
    A = Ktr + lam * np.eye(n)
    c, low = cho_factor(A, check_finite=False)
    alpha = cho_solve((c, low), ytr, check_finite=False)
    return alpha

def krr_predict(Kte: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return Kte @ alpha

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    e = y_true - y_pred
    return float(np.mean(e * e))

def compute_restricted_kernel_matrices(X_train: np.ndarray, X_test: np.ndarray, S, cfg: dict):
    """
    Вычисляет ядерные матрицы для Restricted KRR.
    """
    from .kernels import rbf_kernel, linear_kernel, poly_kernel, matern_kernel

    n, d = X_train.shape

    # Определяем P (pivots)
    if S is None:
        # Случайные индексы по умолчанию
        m = min(500, n)
        indices = np.random.choice(n, size=m, replace=False)
        P = X_train[indices]
    elif isinstance(S, (list, np.ndarray)) and len(S) > 0:
        if isinstance(S[0], (int, np.integer)):
            # S - список/массив индексов
            indices = np.asarray(S, dtype=int)
            P = X_train[indices]
        else:
            # S - массив векторов
            P = np.asarray(S)
            indices = None
    else:
        raise ValueError(f"Неправильный формат S: {type(S)}")

    m = P.shape[0]

    # Определяем функцию ядра и параметры
    kernel_name = cfg["kernel"].lower()

    if kernel_name == "rbf":
        gamma = cfg.get("gamma", 1.0)
        K_XP = rbf_kernel(X_train, P, gamma=gamma)
        Kte_P = rbf_kernel(X_test, P, gamma=gamma)
        K_PP = rbf_kernel(P, P, gamma=gamma)

    elif kernel_name == "linear":
        K_XP = linear_kernel(X_train, P)
        Kte_P = linear_kernel(X_test, P)
        K_PP = linear_kernel(P, P)

    elif kernel_name == "poly":
        degree = cfg.get("degree", 3)
        coef0 = cfg.get("coef0", 1.0)
        gamma = cfg.get("gamma", 1.0)

        K_XP = poly_kernel(X_train, P, degree=degree, coef0=coef0, gamma=gamma)
        Kte_P = poly_kernel(X_test, P, degree=degree, coef0=coef0, gamma=gamma)
        K_PP = poly_kernel(P, P, degree=degree, coef0=coef0, gamma=gamma)

    elif kernel_name == "matern":
        nu = cfg.get("nu", 1.5)
        length_scale = cfg.get("length_scale", 1.0)

        K_XP = matern_kernel(X_train, P, nu=nu, length_scale=length_scale)
        Kte_P = matern_kernel(X_test, P, nu=nu, length_scale=length_scale)
        K_PP = matern_kernel(P, P, nu=nu, length_scale=length_scale)

    else:
        raise ValueError(f"Unknown kernel: {cfg['kernel']}")

    return K_XP, Kte_P, K_PP

def rkrr_exact_alpha(K_XP: np.ndarray, K_PP: np.ndarray, y: np.ndarray, lam: float) -> tuple:
    """
    Решает (K_XP^T K_XP + lam * K_PP) alpha = K_XP^T y через Cholesky.
    Добавляет регуляризацию если матрица не PD.
    """
    A = K_XP.T @ K_XP + lam * K_PP
    b = K_XP.T @ y

    # Проверяем и добавляем регуляризацию
    eps = 1e-8
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            c, low = cho_factor(A, check_finite=False)
            break
        except np.linalg.LinAlgError:
            if attempt == max_attempts - 1:
                raise
            A += eps * np.eye(A.shape[0])
            eps *= 10

    alpha = cho_solve((c, low), b, check_finite=False)

    return alpha

def rkrr_predict(Kte_P: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Предсказание для Restricted KRR"""
    return Kte_P @ alpha
