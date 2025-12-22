import numpy as np
from scipy.special import kv, gamma as gamma_fn

def rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    X2 = (X * X).sum(axis=1)[:, None]
    Y2 = (Y * Y).sum(axis=1)[None, :]
    dist2 = X2 + Y2 - 2.0 * (X @ Y.T)
    return np.exp(-gamma * dist2)

def rbf_kernel_scalar(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    """Скалярная версия RBF ядра для отдельных векторов"""
    return np.exp(-gamma * np.sum((x - y) ** 2))

def linear_kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return X @ Y.T

def poly_kernel(X: np.ndarray, Y: np.ndarray, degree: int = 3, coef0: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    return (gamma * (X @ Y.T) + coef0) ** degree

def matern_kernel(X: np.ndarray, Y: np.ndarray, nu: float = 1.5, length_scale: float = 1.0) -> np.ndarray:
    X2 = (X * X).sum(axis=1)[:, None]
    Y2 = (Y * Y).sum(axis=1)[None, :]
    dist2 = X2 + Y2 - 2.0 * (X @ Y.T)
    dist2 = np.maximum(dist2, 0.0)
    r = np.sqrt(dist2) / max(length_scale, 1e-12)

    if nu == 0.5:
        return np.exp(-r)
    if nu == 1.5:
        t = np.sqrt(3.0) * r
        return (1.0 + t) * np.exp(-t)
    if nu == 2.5:
        t = np.sqrt(5.0) * r
        return (1.0 + t + (t * t) / 3.0) * np.exp(-t)

    t = np.sqrt(2.0 * nu) * r
    out = np.zeros_like(r)
    mask = (r > 0)
    coef = (2.0 ** (1.0 - nu)) / gamma_fn(nu)
    out[~mask] = 1.0
    out[mask] = coef * (t[mask] ** nu) * kv(nu, t[mask])
    return out

def compute_kernel_matrices(Xtr, Xte, cfg):
    k = cfg["kernel"].lower()
    if k == "rbf":
        Ktr = rbf_kernel(Xtr, Xtr, gamma=cfg["gamma"])
        Kte = rbf_kernel(Xte, Xtr, gamma=cfg["gamma"])
        return Ktr, Kte
    if k == "linear":
        return linear_kernel(Xtr, Xtr), linear_kernel(Xte, Xtr)
    if k == "poly":
        return (
            poly_kernel(Xtr, Xtr, degree=cfg.get("degree",3), coef0=cfg.get("coef0",1.0), gamma=cfg.get("gamma",1.0)),
            poly_kernel(Xte, Xtr, degree=cfg.get("degree",3), coef0=cfg.get("coef0",1.0), gamma=cfg.get("gamma",1.0)),
        )
    if k == "matern":
        return (
            matern_kernel(Xtr, Xtr, nu=cfg.get("nu",1.5), length_scale=cfg.get("length_scale",1.0)),
            matern_kernel(Xte, Xtr, nu=cfg.get("nu",1.5), length_scale=cfg.get("length_scale",1.0)),
        )
    raise ValueError(f"Unknown kernel: {cfg['kernel']}")

class KernelMatrix:
    """Абстрактная ядерная матрица для RPCholesky"""

    def __init__(self, X: np.ndarray, cfg: dict):
        """
        Parameters
        ----------
        X : np.ndarray [n, d]
            Данные
        cfg : dict
            Конфигурация ядра, например:
            {"kernel": "rbf", "gamma": 0.1}
            {"kernel": "poly", "degree": 3, "gamma": 1.0, "coef0": 1.0}
            {"kernel": "matern", "nu": 1.5, "length_scale": 1.0}
        """
        self.X = X
        self.n = X.shape[0]
        self.shape = (self.n, self.n)
        self.cfg = cfg.copy()  # Копируем конфиг

        # Определяем функцию ядра
        kernel_name = cfg["kernel"].lower()
        if kernel_name == "rbf":
            self.kernel_func = rbf_kernel
        elif kernel_name == "linear":
            self.kernel_func = linear_kernel
        elif kernel_name == "poly":
            self.kernel_func = poly_kernel
        elif kernel_name == "matern":
            self.kernel_func = matern_kernel
        else:
            raise ValueError(f"Unknown kernel: {cfg['kernel']}")

    def get_row(self, idx: int) -> np.ndarray:
        """Вычисляет одну строку ядерной матрицы"""
        X_i = self.X[idx:idx+1]
        return self.kernel_func(X_i, self.X, **self._extract_params())[0]

    def get_rows(self, indices: np.ndarray) -> np.ndarray:
        """Вычисляет несколько строк одновременно"""
        return self.kernel_func(self.X[indices], self.X, **self._extract_params())

    def _extract_params(self) -> dict:
        """Извлекает параметры ядра из cfg"""
        kernel_name = self.cfg["kernel"].lower()
        params = {}

        if kernel_name == "rbf":
            params["gamma"] = self.cfg.get("gamma", 1.0)
        elif kernel_name == "poly":
            params["degree"] = self.cfg.get("degree", 3)
            params["coef0"] = self.cfg.get("coef0", 1.0)
            params["gamma"] = self.cfg.get("gamma", 1.0)
        elif kernel_name == "matern":
            params["nu"] = self.cfg.get("nu", 1.5)
            params["length_scale"] = self.cfg.get("length_scale", 1.0)
        # linear не имеет параметров

        return params

    def diag(self) -> np.ndarray:
        """Диагональ ядерной матрицы"""
        params = self._extract_params()
        kernel_name = self.cfg["kernel"].lower()

        if kernel_name == "rbf":
            return np.ones(self.n)
        elif kernel_name == "linear":
            return np.sum(self.X**2, axis=1)
        elif kernel_name == "poly":
            gamma = params.get("gamma", 1.0)
            coef0 = params.get("coef0", 1.0)
            degree = params.get("degree", 3)
            norms = np.sum(self.X**2, axis=1)
            return (gamma * norms + coef0) ** degree
        elif kernel_name == "matern":
            return np.ones(self.n)
        else:
            # Общий случай
            diag = np.zeros(self.n)
            for i in range(self.n):
                diag[i] = self.kernel_func(
                    self.X[i:i+1], self.X[i:i+1], **params
                )[0, 0]
            return diag
