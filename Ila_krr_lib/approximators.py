import numpy as np
from scipy.linalg import solve_triangular, cho_factor, cho_solve, eigh
from sklearn.cluster import KMeans

def randomized_nystrom_eig(A: np.ndarray, ell: int, seed: int = 0, eps: float = 1e-12):
    """
    Build rank-ell randomized Nyström approximation of A (psd),
    return eigenpairs A_hat = U diag(lam) U^T, with U orthonormal (n×ell), lam desc.
    Implements A_hat = (AΩ)(Ω^T A Ω)^†(AΩ)^T from the paper.
    """
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    Omega = rng.normal(size=(n, ell))

    Y = A @ Omega
    W = Omega.T @ Y
    W = 0.5 * (W + W.T)

    w_vals, w_vecs = np.linalg.eigh(W)
    w_vals = np.clip(w_vals, eps, None)
    Winvhalf = w_vecs @ np.diag(1.0 / np.sqrt(w_vals)) @ w_vecs.T

    B = Y @ Winvhalf

    C = B.T @ B
    C = 0.5 * (C + C.T)
    lam, V = np.linalg.eigh(C)

    idx = np.argsort(lam)[::-1]
    lam = lam[idx]
    V = V[:, idx]
    lam = np.clip(lam, eps, None)

    U = B @ (V @ np.diag(1.0 / np.sqrt(lam)))

    U, _ = np.linalg.qr(U)

    return U, lam

import warnings
warnings.filterwarnings("ignore")

from .kernels import KernelMatrix

def rpcholesky(A: KernelMatrix, rank: int = 100) -> tuple:
    """
    RPCholesky для KernelMatrix объекта.
    Добавлена регуляризация для обеспечения PD и избежание дубликатов.
    """
    n = A.n

    # Инициализация
    diags = A.diag()
    # Добавляем минимальную регуляризацию к диагонали для устойчивости
    diags = np.maximum(diags, 1e-8)
    total_trace = np.sum(diags)
    G = np.zeros((rank, n))
    selected_idx = []
    selected_set = set()  # Для избежания дубликатов

    for k in range(rank):
        # 1. Выбор точки с вероятностью пропорциональной диагонали, исключая уже выбранные
        if total_trace < 1e-12:
            break

        available = np.ones(n, dtype=bool)
        available[list(selected_set)] = False
        if not np.any(available):
            break

        probs = diags * available
        probs_sum = np.sum(probs)
        if probs_sum < 1e-12:
            break
        probs = probs / probs_sum

        idx = np.random.choice(n, p=probs)
        selected_idx.append(idx)
        selected_set.add(idx)

        # 2. Вычисляем ВСЮ строку ядра
        row = A.get_row(idx)

        # 3. Ортогонализация Грамма-Шмидта
        if k > 0:
            row = row - G[:k, idx].T @ G[:k, :]

        # 4. Нормировка с регуляризацией
        scale = np.sqrt(max(diags[idx], 1e-8))
        G[k, :] = row / scale

        # 5. Обновление остатков
        diags = diags - G[k, :]**2
        diags = np.maximum(diags, 0)  # Не даем уйти в отрицательные
        total_trace = np.sum(diags)

    # Обрезаем, если закончились точки
    actual_rank = len(selected_idx)
    if actual_rank < rank:
        G = G[:actual_rank, :]

    # Проверяем на дубликаты
    if len(set(selected_idx)) != len(selected_idx):
        print(f"Warning: Duplicates in selected_idx: {selected_idx}")

    return G, selected_idx

class RecursiveNystromWrapper(object):
    def __init__(self, X, gamma):
        self.X = X
        self.gamma = gamma

    def __call__(self, X_idx, Y_idx=None):
        from .kernels import rbf_kernel
        if Y_idx is None:
            # Diagonal
            K = rbf_kernel(self.X[X_idx.flatten()], self.X[X_idx.flatten()], gamma=self.gamma)
            return np.diag(K).reshape(-1, 1)
        else:
            return rbf_kernel(self.X[X_idx.flatten()], self.X[Y_idx.flatten()], gamma=self.gamma)

def recursiveNystrom(X, n_components: int, kernel_func, random_state=None, lmbda_0=0, return_leverage_score=False):
    # set up parameters
    rng = np.random.default_rng(random_state)
    n_oversample = np.log(n_components)
    k = np.ceil(n_components / (4 * n_oversample)).astype(int)
    n_levels = np.ceil(np.log(X.shape[0] / n_components) / np.log(2)).astype(int)
    perm = rng.permutation(X.shape[0])

    # set up sizes for recursive levels
    size_list = [X.shape[0]]
    for l in range(1, n_levels+1):
        size_list += [np.ceil(size_list[l - 1] / 2).astype(int)]

    # indices of points selected at previous level of recursion
    # at the base level it's just a uniform sample of ~ n_component points
    sample = np.arange(size_list[-1])
    indices = perm[sample]
    weights = np.ones((indices.shape[0],))

    # we need the diagonal of the whole kernel matrix, so compute upfront
    k_diag = kernel_func(np.arange(X.shape[0]).reshape(-1, 1))

    # Main recursion, unrolled for efficiency
    for l in reversed(range(n_levels)):
        # indices of current uniform sample
        current_indices = perm[:size_list[l]]
        # build sampled kernel

        # all rows and sampled columns
        KS = kernel_func(current_indices.reshape(-1, 1), indices.reshape(-1, 1))
        SKS = KS[sample, :] # sampled rows and sampled columns

        # optimal lambda for taking O(k log(k)) samples
        if k >= SKS.shape[0]:
            # for the rare chance we take less than k samples in a round
            lmbda = 1e-6
            # don't set to exactly 0 to avoid stability issues
        else:
            # eigenvalues equal roughly the number of points per cluster, maybe this should scale with n?
            # can be interpret as the zoom level
            lmbda = (np.sum(np.diag(SKS) * (weights ** 2))
                    - np.sum(eigh(SKS * weights[:,None] * weights[None,:], subset_by_index=(SKS.shape[0]-k, SKS.shape[0]-1))[0]))/k
        lmbda = np.maximum(lmbda_0 * SKS.shape[0], lmbda)
        if lmbda == lmbda_0 * SKS.shape[0]:
            print("Set lambda to %d." % lmbda)

        # compute and sample by lambda ridge leverage scores
        R = np.linalg.inv(SKS + np.diag(lmbda * weights ** (-2)))
        R = np.matmul(KS, R)
        if l != 0:
            leverage_score = np.minimum(1.0, n_oversample * (1 / lmbda) * np.maximum(+0.0, (
                    k_diag[current_indices, 0] - np.sum(R * KS, axis=1))))
            # on intermediate levels, we independently sample each column
            # by its leverage score. the sample size is n_components in expectation
            sample = np.where(rng.uniform(size=size_list[l]) < leverage_score)[0]
            # with very low probability, we could accidentally sample no
            # columns. In this case, just take a fixed size uniform sample
            if sample.size == 0:
                leverage_score[:] = n_components / size_list[l]
                sample = rng.choice(size_list[l], size=n_components, replace=False)
            weights = np.sqrt(1. / leverage_score[sample])

        else:
            leverage_score = np.minimum(1.0, (1 / lmbda) * np.maximum(+0.0, (
                    k_diag[current_indices, 0] - np.sum(R * KS, axis=1))))
            p = leverage_score/leverage_score.sum()

            sample = rng.choice(X.shape[0], size=n_components, replace=False, p=p)
        indices = perm[sample]

    if return_leverage_score:
        return indices, leverage_score[np.argsort(perm)]
    else:
        return indices

def select_control_points_rls(X: np.ndarray, m: int, lambda_reg: float = 0.0, gamma: float = None) -> np.ndarray:
    """
    Select m control points using recursive Nystrom method for leverage scores.
    """
    from .utilities import median_heuristic_gamma

    n = len(X)
    if m >= n:
        return np.arange(n)

    if gamma is None:
        gamma = median_heuristic_gamma(X)
    kernel_func = RecursiveNystromWrapper(X, gamma)
    X_idx = np.arange(n).reshape(-1, 1)
    indices = recursiveNystrom(X_idx, m, kernel_func, random_state=42, lmbda_0=lambda_reg)
    return indices

def block_rpcholesky(A, rank: int = 100, block_size: int = 20) -> tuple:
    """
    Блочный RPCholesky для KernelMatrix.
    """
    n = A.n

    diags = A.diag()
    total_trace = np.sum(diags)
    G = np.zeros((rank, n))
    selected_idx = []

    k = 0
    while k < rank:
        # Размер текущего блока
        b = min(block_size, rank - k)

        # Проверка на вырожденность
        if total_trace < 1e-12:
            break

        # 1. Выбор кандидатов
        probs = diags / total_trace
        candidates = np.random.choice(n, size=b, p=probs, replace=False)

        # 2. Вычисляем ВСЕ строки кандидатов сразу
        rows = A.get_rows(candidates)  # (b, n)

        # 3. Вычисляем подматрицу для ортогонализации
        K_block = rows[:, candidates]  # (b, b)

        # 4. Ортогонализация относительно предыдущих
        if k > 0:
            rows = rows - G[:k, candidates].T @ G[:k, :]
            K_block = K_block - G[:k, candidates].T @ G[:k, candidates]

        # 5. Холецкого блока
        try:
            L_block = np.linalg.cholesky(K_block + 1e-12 * np.eye(b))
        except np.linalg.LinAlgError:
            evals, evecs = np.linalg.eigh(K_block)
            evals = np.maximum(evals, 0)
            L_block = (evecs * np.sqrt(evals)).T

        # 6. Обновление
        G_new = solve_triangular(L_block, rows, lower=True)
        G[k:k+b, :] = G_new
        selected_idx.extend(candidates.tolist())

        # 7. Обновление остатков
        diags = diags - np.sum(G_new**2, axis=0)
        diags = np.clip(diags, 0, None)
        total_trace = np.sum(diags)

        k += b

    # Обрезаем до фактического ранга
    if k < rank:
        G = G[:k, :]

    return G, selected_idx

def select_pivots_kmeans(X: np.ndarray, n_pivots: int, seed: int = 0, max_iter: int = 100, tol: float = 1e-4) -> np.ndarray:
    """
    Select pivots using k-means clustering with k-means++ initialization for better quality and convergence.
    Uses scikit-learn's optimized KMeans implementation.
    Returns indices of selected pivots.
    """
    # Use scikit-learn's KMeans for faster computation
    kmeans = KMeans(n_clusters=n_pivots, init='k-means++', n_init=1, max_iter=max_iter, tol=tol, random_state=seed)
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    # For each cluster, find the closest point to centroid
    pivots = []
    for i in range(n_pivots):
        cluster_mask = labels == i
        if np.any(cluster_mask):
            cluster_points = X[cluster_mask]
            dist_sq = np.sum((cluster_points - centroids[i])**2, axis=1)
            closest_idx = np.argmin(dist_sq)
            global_idx = np.where(cluster_mask)[0][closest_idx]
            pivots.append(global_idx)
        else:
            # Fallback: should not happen with sklearn, but just in case
            pivots.append(np.random.choice(len(X)))

    return np.array(pivots)

# KRILL method
import time
from scipy.sparse import csr_matrix

def sparse_sign_embedding(d: int, N: int, zeta: int, seed: int = 0) -> csr_matrix:
    rng = np.random.default_rng(seed)
    zeta = int(max(1, zeta))
    d = int(d); N = int(N)

    nnz = zeta * N
    rows = np.empty(nnz, dtype=np.int32)
    cols = np.empty(nnz, dtype=np.int32)
    vals = np.empty(nnz, dtype=np.float64)

    scale = 1.0 / np.sqrt(zeta)
    p = 0
    signs = np.array([-1.0, 1.0], dtype=np.float64)
    for j in range(N):
        r = rng.choice(d, size=zeta, replace=False)
        s = rng.choice(signs, size=zeta, replace=True)
        rows[p:p+zeta] = r
        cols[p:p+zeta] = j
        vals[p:p+zeta] = scale * s
        p += zeta

    return csr_matrix((vals, (rows, cols)), shape=(d, N))

def krill_restricted_solve(Ktr: np.ndarray, ytr: np.ndarray, lam: float, k_centers: int,
                          seed: int = 0, tol: float = 1e-6, max_iter: int = 500):
    from .solvers import pcg  # Import pcg from solvers

    N = int(Ktr.shape[0])
    k = int(min(k_centers, N))
    rng = np.random.default_rng(seed)

    t_sel0 = time.perf_counter()
    S = rng.choice(N, size=k, replace=False).astype(int)
    t_sel1 = time.perf_counter()
    time_selectS = t_sel1 - t_sel0

    AS = Ktr[:, S]
    KSS = Ktr[np.ix_(S, S)]

    epsmach = np.finfo(np.float64).eps
    H = lam * KSS.copy()
    H.flat[::k+1] += N * epsmach * float(np.trace(KSS))

    d = int(2 * k)
    zeta = int(np.ceil(np.log(k + 1.0)))

    t0 = time.perf_counter()
    Phi = sparse_sign_embedding(d=d, N=N, zeta=zeta, seed=seed + 12345)
    B = Phi @ AS
    P = (B.T @ B) + H
    cF = cho_factor(P, lower=True, check_finite=False)
    t1 = time.perf_counter()
    time_precond = t1 - t0

    def M_apply(v: np.ndarray) -> np.ndarray:
        return AS.T @ (AS @ v) + H @ v

    def Pinv_apply(v: np.ndarray) -> np.ndarray:
        return cho_solve(cF, v, check_finite=False)

    rhs = AS.T @ ytr
    beta, iters, rel_res = pcg(M_apply, rhs, M_apply=Pinv_apply, tol=tol, max_iter=max_iter)

    t2 = time.perf_counter()
    time_solve = t2 - t1

    return {
        "S": S,
        "beta": beta,
        "iters": int(iters),
        "rel_res": float(rel_res),
        "time_selectS": float(time_selectS),
        "time_precond": float(time_precond),
        "time_solve": float(time_solve),
        "time_total": float(time_selectS + time_precond + time_solve),
        "k": int(k),
        "d": int(d),
        "zeta": int(zeta),
    }

# AFN method
def _rkhs_dist2_from_gram(K: np.ndarray, j: int) -> np.ndarray:
    diag = np.diag(K)
    return np.maximum(diag + diag[j] - 2.0 * K[:, j], 0.0)

def fps_indices_from_gram(K: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    n = K.shape[0]
    k = int(min(k, n))
    rng = np.random.default_rng(seed)

    first = int(rng.integers(0, n))
    chosen = [first]
    min_d2 = _rkhs_dist2_from_gram(K, first)

    for _ in range(1, k):
        nxt = int(np.argmax(min_d2))
        chosen.append(nxt)
        d2_new = _rkhs_dist2_from_gram(K, nxt)
        min_d2 = np.minimum(min_d2, d2_new)

    return np.array(chosen, dtype=int)

def fsai_from_schur(S: np.ndarray, K_rr_for_knn: np.ndarray, w: int = 20, jitter: float = 1e-12) -> csr_matrix:
    m = S.shape[0]
    w = int(max(1, w))
    diagK = np.diag(K_rr_for_knn)

    rows, cols, vals = [], [], []

    for i in range(m):
        if i == 0 or w == 1:
            si = np.array([i], dtype=int)
        else:
            d2 = np.maximum(diagK[:i] + diagK[i] - 2.0 * K_rr_for_knn[i, :i], 0.0)
            nn = min(w - 1, i)
            idx = np.argpartition(d2, nn - 1)[:nn]
            idx = np.sort(idx.astype(int))
            si = np.concatenate([idx, np.array([i], dtype=int)])

        Ssub = S[np.ix_(si, si)].copy()
        Ssub.flat[::Ssub.shape[0] + 1] += jitter

        e = np.zeros(len(si))
        e[-1] = 1.0

        u = np.linalg.solve(Ssub, e)
        denom = float(u[-1])
        if denom <= 0:
            Ssub.flat[::Ssub.shape[0] + 1] += 100.0 * jitter
            u = np.linalg.solve(Ssub, e)
            denom = float(u[-1])

        g = u / np.sqrt(denom)

        rows.extend([i] * len(si))
        cols.extend(si.tolist())
        vals.extend(g.tolist())

    return csr_matrix((vals, (rows, cols)), shape=(m, m))

def build_afn_preconditioner(K: np.ndarray, mu: float, k_landmarks: int, w: int = 20, seed: int = 0,
                             kmax: int = 2000, jitter: float = 1e-12):
    t0 = time.perf_counter()
    n = K.shape[0]
    k = int(min(k_landmarks, n - 1, kmax))
    if k <= 0:
        def apply_Pinv(r):
            return r / (np.diag(K) + mu + 1e-30)
        return apply_Pinv, {"k": 0, "w": int(w), "time_total": time.perf_counter() - t0}

    # 1) landmarks
    t1 = time.perf_counter()
    S = fps_indices_from_gram(K, k=k, seed=seed)
    mask = np.ones(n, dtype=bool)
    mask[S] = False
    R = np.where(mask)[0]
    time_landmarks = time.perf_counter() - t1

    # blocks
    K_ss = K[np.ix_(S, S)]
    K_sr = K[np.ix_(S, R)]
    K_rs = K_sr.T
    K_rr = K[np.ix_(R, R)]

    # 2) Cholesky of (K_ss + mu I)
    t2 = time.perf_counter()
    A11 = K_ss + mu * np.eye(k)
    L = np.linalg.cholesky(A11 + jitter * np.eye(k))
    time_chol = time.perf_counter() - t2

    # 3) Schur complement
    t3 = time.perf_counter()
    Y = solve_triangular(L, K_sr, lower=True, check_finite=False)
    X = solve_triangular(L.T, Y, lower=False, check_finite=False)
    Schur = (K_rr + mu * np.eye(len(R))) - (K_rs @ X)
    Schur.flat[::Schur.shape[0] + 1] += jitter
    time_schur = time.perf_counter() - t3

    # 4) FSAI
    t4 = time.perf_counter()
    G = fsai_from_schur(Schur, K_rr_for_knn=K_rr, w=w, jitter=jitter)
    time_fsai = time.perf_counter() - t4

    LinvKsr = solve_triangular(L, K_sr, lower=True, check_finite=False)

    def apply_Pinv(r: np.ndarray) -> np.ndarray:
        rS = r[S]
        rR = r[R]

        u1 = solve_triangular(L, rS, lower=True, check_finite=False)
        v1 = solve_triangular(L.T, u1, lower=False, check_finite=False)

        rhs2 = rR - (K_rs @ v1)

        u2 = G.dot(rhs2)
        zR = G.T.dot(u2)

        u1_corr = u1 - (LinvKsr @ zR)
        zS = solve_triangular(L.T, u1_corr, lower=False, check_finite=False)

        z = np.zeros_like(r)
        z[S] = zS
        z[R] = zR
        return z

    info = {
        "k": int(k), "w": int(w),
        "S_size": int(len(S)), "R_size": int(len(R)),
        "time_landmarks": float(time_landmarks),
        "time_chol": float(time_chol),
        "time_schur": float(time_schur),
        "time_fsai": float(time_fsai),
        "time_total": float(time.perf_counter() - t0),
    }
    return apply_Pinv, info

def select_control_points_rls_recursive(X: np.ndarray, m: int, lambda_reg: float = 0.01, seed: int = 42) -> np.ndarray:
    """
    Select m control points using recursive leverage score sampling based on Musco and Musco's
    "Recursive Sampling for the Nystrom Method". This provides a more efficient way to select
    points by maintaining a low-rank approximation and updating leverage scores recursively.
    """
    from .utilities import median_heuristic_gamma
    from .kernels import rbf_kernel

    n, d = X.shape
    gamma = median_heuristic_gamma(X)
    rng = np.random.default_rng(seed)

    if m >= n:
        return np.arange(n)

    # Start with a random point
    selected = [rng.choice(n)]
    remaining = set(range(n)) - set(selected)

    # Compute initial kernel matrix for selected points
    K_sel = rbf_kernel(X[selected], X[selected], gamma=gamma)
    A_sel = K_sel + lambda_reg * np.eye(len(selected))

    # Initialize low-rank approximation
    U = np.eye(len(selected))
    Sigma = np.linalg.inv(A_sel)

    for _ in range(1, m):
        # Compute cross-kernel between selected and remaining points
        K_cross = rbf_kernel(X[selected], X[list(remaining)], gamma=gamma)  # (m_current, n_remaining)

        # Compute approximate leverage scores for remaining points
        # Using the current low-rank approximation: scores ≈ diag(K_cross.T @ Sigma @ K_cross)
        temp = Sigma @ K_cross  # (m_current, n_remaining)
        scores = np.sum(K_cross * temp, axis=0)  # (n_remaining,)

        # Select the point with highest score
        max_idx = np.argmax(scores)
        new_point = list(remaining)[max_idx]

        # Add to selected
        selected.append(new_point)
        remaining.remove(new_point)

        # Update the approximation recursively
        # New row and column for the kernel matrix
        k_new = rbf_kernel(X[selected[:-1]], X[[new_point]], gamma=gamma).flatten()  # (m_current,)
        k_new_new = rbf_kernel(X[[new_point]], X[[new_point]], gamma=gamma)[0, 0]  # scalar

        # Update A_sel
        A_new = np.zeros((len(selected), len(selected)))
        A_new[:-1, :-1] = A_sel
        A_new[:-1, -1] = k_new
        A_new[-1, :-1] = k_new
        A_new[-1, -1] = k_new_new + lambda_reg

        # Update Sigma using Sherman-Morrison for rank-1 update
        # A_new = A_old + u v^T, where u = v = [k_new; k_new_new + lambda_reg - lambda_reg] wait, better way
        # Since adding a row/column, use block inverse update
        m_curr = len(selected) - 1
        temp = Sigma @ k_new  # (m_curr,)
        denom = k_new_new + lambda_reg - k_new @ temp
        if abs(denom) < 1e-12:
            denom = np.sign(denom) * 1e-12

        # Update Sigma
        Sigma_new = np.zeros((m_curr + 1, m_curr + 1))
        Sigma_new[:-1, :-1] = Sigma + np.outer(temp, temp) / denom
        Sigma_new[:-1, -1] = -temp / denom
        Sigma_new[-1, :-1] = -temp / denom
        Sigma_new[-1, -1] = 1.0 / denom

        Sigma = Sigma_new
        A_sel = A_new

    return np.array(selected)
