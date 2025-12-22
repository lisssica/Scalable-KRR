import numpy as np

def standardize_features(X: np.ndarray):
    mu = X.mean(axis=0, keepdims=True)
    sig = X.std(axis=0, keepdims=True)
    sig = np.where(sig < 1e-12, 1.0, sig)
    return (X - mu) / sig, mu, sig

def load_ccpp(seed: int = 0, n_sample: int or None = None, standardize: bool = True):
    """
    Combined Cycle Power Plant (CCPP), UCI id=294.
    Features: 4 continuous (AT, V, AP, RH). Target: PE (energy output).
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        raise ImportError("ucimlrepo is required for loading real datasets. Install with: pip install ucimlrepo")

    rng = np.random.default_rng(seed)
    data = fetch_ucirepo(id=294)
    X = data.data.features.to_numpy().astype(float)
    y = data.data.targets.to_numpy().ravel().astype(float)

    if n_sample is not None and X.shape[0] > n_sample:
        idx = rng.choice(X.shape[0], size=n_sample, replace=False)
        X = X[idx]
        y = y[idx]

    if standardize:
        X, _, _ = standardize_features(X)

    return X, y

def load_protein(seed: int = 0, n_sample: int or None = None, standardize: bool = True):
    """
    Physicochemical Properties of Protein Tertiary Structure, UCI id=265.
    Features: 9 continuous. Target: RMSD (Root Mean Square Deviation).
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    # Load from UCI repository online
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv"
    df = pd.read_csv(url)
    X = df[["F1","F2","F3","F4","F5","F6","F7","F8","F9"]].to_numpy().astype(float)
    y = df["RMSD"].to_numpy().astype(float)

    if n_sample is not None and X.shape[0] > n_sample:
        idx = rng.choice(X.shape[0], size=n_sample, replace=False)
        X = X[idx]
        y = y[idx]

    if standardize:
        X, _, _ = standardize_features(X)

    return X, y


def load_real_dataset(name: str, seed: int = 0, n_sample: int or None = None):
    name = name.lower()
    if name in ["ccpp", "powerplant", "combined_cycle_power_plant"]:
        return load_ccpp(seed=seed, n_sample=n_sample, standardize=True)
    elif name in ["protein", "protein_tertiary_structure"]:
        return load_protein(seed=seed, n_sample=n_sample, standardize=True)
    raise ValueError("Unknown real dataset name (currently supported: CCPP, Protein)")

def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_tr = int((1.0 - test_ratio) * n)
    tr = idx[:n_tr]
    te = idx[n_tr:]
    return X[tr], y[tr], X[te], y[te]

def standardize_train_test(Xtr: np.ndarray, Xte: np.ndarray):
    mu = Xtr.mean(axis=0, keepdims=True)
    sig = Xtr.std(axis=0, keepdims=True)
    sig = np.where(sig < 1e-12, 1.0, sig)
    return (Xtr - mu) / sig, (Xte - mu) / sig
