import time
import numpy as np
import pandas as pd
import tracemalloc
import gc
import matplotlib.pyplot as plt
from nla_krr_lib import datasets, kernels, utilities, approximators, preconditioners, solvers

def run_experiment(Xtr, Xte, ytr, yte, cfg, method, ell=None, deff_info=None, seed=0):
    """
    Run one method with memory and time tracking.
    Returns dict with results.
    """
    lam = cfg["lam"]
    n_train = Xtr.shape[0]

    # Start memory tracking
    tracemalloc.start()
    start_time = time.perf_counter()

    try:
        if method in ["restricted_rpchol", "restricted_block_rpchol", "restricted_rls", "restricted_k_means"]:
            # For restricted methods
            if method == "restricted_block_nystrom":
                alpha, control_indices = solvers.restricted_block_nystrom_krr(Xtr, ytr, cfg, m=ell, q=10)
                start_pred_time = time.perf_counter()
                y_pred = solvers.predict_restricted_block_nystrom(Xte, Xtr, alpha, control_indices, cfg)
                pred_time = time.perf_counter() - start_pred_time
                iters = 0
                rel_res = np.nan
            else:
                K = kernels.KernelMatrix(Xtr, cfg)
                if method == "restricted_rpchol":
                    _, S = approximators.rpcholesky(K, rank=ell)
                elif method == "restricted_block_rpchol":
                    _, S = approximators.block_rpcholesky(K, rank=ell, block_size=ell // 5)
                elif method == "restricted_rls":
                    S = approximators.select_control_points_rls(Xtr, ell, lambda_reg=lam)
                elif method == "restricted_k_means":
                    S = approximators.select_pivots_kmeans(Xtr, n_pivots=ell, seed=seed)

                K_XP, Kte_P, K_PP = solvers.compute_restricted_kernel_matrices(Xtr, Xte, S, cfg)
                alpha = solvers.rkrr_exact_alpha(K_XP, K_PP, ytr, lam)
                start_pred_time = time.perf_counter()
                y_pred = solvers.rkrr_predict(Kte_P, alpha)
                pred_time = time.perf_counter() - start_pred_time
                iters = 0
                rel_res = np.nan

        else:
            # For full methods
            Ktr, Kte = kernels.compute_kernel_matrices(Xtr, Xte, cfg)
            if deff_info is None:
                deff_info = utilities.effective_dimension(Ktr, mu=lam)

            A_apply = lambda v: Ktr @ v + lam * v

            if method == "full_exact":
                alpha = solvers.krr_exact_solve(Ktr, ytr, lam)
                iters = 0
                rel_res = 0.0

            elif method == "cg":
                alpha, iters, rel_res = solvers.pcg(A_apply, ytr, M_apply=None, tol=cfg["tol"], max_iter=cfg["max_iter"])

            elif method == "nystrom_pcg":
                U, lam_hat = approximators.randomized_nystrom_eig(Ktr, ell=ell, seed=seed)
                apply_Pinv = preconditioners.make_nystrom_precond_apply(U, lam_hat, mu=lam)
                alpha, iters, rel_res = solvers.pcg(A_apply, ytr, M_apply=apply_Pinv, tol=cfg["tol"], max_iter=cfg["max_iter"])

            elif method == "rpchol_pcg":
                L, pivots, diag_res = preconditioners.rp_cholesky_factor(Ktr, ell=ell, seed=seed)
                apply_Pinv = preconditioners.make_rpchol_precond_apply(L, lam=lam)
                alpha, iters, rel_res = solvers.pcg(A_apply, ytr, M_apply=apply_Pinv, tol=cfg["tol"], max_iter=cfg["max_iter"])

            elif method == "krill":
                k_centers = int(ell)
                res = approximators.krill_restricted_solve(Ktr, ytr, lam=lam, k_centers=k_centers, seed=seed, tol=cfg["tol"], max_iter=cfg["max_iter"])
                S = res["S"]
                beta = res["beta"]
                alpha = np.zeros(n_train)
                alpha[S] = beta
                iters = res["iters"]
                rel_res = res["rel_res"]

            elif method == "afn_pcg":
                w = cfg.get("afn_w", 20)
                kmax = cfg.get("afn_kmax", 2000)
                jitter = cfg.get("afn_jitter", 1e-12)
                apply_Pinv, afn_info = approximators.build_afn_preconditioner(Ktr, mu=lam, k_landmarks=int(ell), w=w, seed=seed, kmax=kmax, jitter=jitter)
                alpha, iters, rel_res = solvers.pcg(A_apply, ytr, M_apply=apply_Pinv, tol=cfg["tol"], max_iter=cfg["max_iter"])

            else:
                raise ValueError(f"Unknown method: {method}")

            start_pred_time = time.perf_counter()
            y_pred = solvers.krr_predict(Kte, alpha)
            pred_time = time.perf_counter() - start_pred_time

        test_mse = solvers.mse(yte, y_pred)
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Free memory
        gc.collect()

        result = {
            "dataset": cfg["dataset"],
            "kernel": cfg["kernel"],
            "n_train": n_train,
            "n_test": Xte.shape[0],
            "lam": lam,
            "method": method,
            "ell": ell if ell is not None else 0,  # ell is always provided for methods that need it
            "tol": cfg["tol"],
            "max_iter": cfg["max_iter"],
            "iters": iters,
            "rel_res": rel_res,
            "test_mse": test_mse,
            "time_total": end_time - start_time,
            "pred_time": pred_time,
            "peak_memory": peak / 1024 / 1024,  # MB
            "seed": seed,
        }

        # Add kernel params
        for k in ["gamma", "degree", "coef0", "nu", "length_scale"]:
            if k in cfg:
                result[k] = cfg[k]

        return result

    except Exception as e:
        tracemalloc.stop()
        gc.collect()
        raise e

def main():
    datasets_list = [ "protein", "ccpp"]

    all_results = []

    for dataset_name in datasets_list:
        cfg_base = {
            "dataset": dataset_name,
            "test_ratio": 0.2,
            "seed": 0,
            "lam": 1.0,
            "kernel": "rbf",
            "gamma": 1.0,
            "tol": 1e-6,
            "max_iter": 2000,
            "afn_w": 20,
            "afn_kmax": 2000,
            "afn_jitter": 1e-12,
        }

        if dataset_name == "protein":
            n_sample = 10000
        elif dataset_name == "ccpp":
            n_sample = 4000
        else:
            n_sample = None
        X, y = datasets.load_real_dataset(dataset_name, seed=cfg_base["seed"], n_sample=n_sample)
        Xtr, ytr, Xte, yte = datasets.train_test_split(X, y, test_ratio=cfg_base["test_ratio"], seed=cfg_base["seed"])

        # Compute effective rank for each lambda and kernel
        lam_grid = [1.0, 1e-1, 1e-2]
        kernel_grid = [
            {"kernel": "rbf", "gamma": utilities.median_heuristic_gamma(Xtr, seed=cfg_base["seed"])},
            {"kernel": "matern", "nu": 1.5, "length_scale": utilities.median_length_scale(Xtr, seed=cfg_base["seed"])},
            {"kernel": "poly", "degree": 3, "coef0": 1.0, "gamma": 1.0 / max(Xtr.shape[1], 1)},
        ]

        # Compute ell and deff_info for each combination
        ell_grid = {}
        deff_grid = {}
        for lam in lam_grid:
            for kcfg in kernel_grid:
                cfg_temp = cfg_base.copy()
                cfg_temp["lam"] = lam
                cfg_temp.update(kcfg)
                Ktr, _ = kernels.compute_kernel_matrices(Xtr, Xte, cfg_temp)
                deff_info = utilities.effective_dimension(Ktr, mu=lam)
                ell = utilities.suggest_ell_from_deff(deff_info["d_eff_est"])
                ell = int(min(ell, Ktr.shape[0] - 2))
                key = (lam, kcfg["kernel"])
                ell_grid[key] = ell
                deff_grid[key] = deff_info
                print(f"Dataset: {dataset_name}, Lambda: {lam}, Kernel: {kcfg['kernel']}, d_eff: {deff_info['d_eff_est']:.2f}, ell: {ell}")

        # Methods
        methods = ["full_exact", "cg", "nystrom_pcg", "rpchol_pcg", "restricted_rpchol", "restricted_block_rpchol", "restricted_rls", "restricted_k_means", "krill", "afn_pcg"]

        results = []

        for lam in lam_grid:
            for kcfg in kernel_grid:
                cfg = cfg_base.copy()
                cfg["lam"] = lam
                cfg.update(kcfg)
                key = (lam, kcfg["kernel"])
                ell = ell_grid[key]

                for method in methods:
                    try:
                        deff_info = deff_grid[key] if method not in ["restricted_rpchol", "restricted_block_rpchol", "restricted_rls", "restricted_k_means"] else {}
                        row = run_experiment(Xtr, Xte, ytr, yte, cfg, method, ell=ell, deff_info=deff_info, seed=cfg["seed"])
                        results.append(row)
                        print(f"Completed: {dataset_name}, {method}, lam={lam}, kernel={kcfg['kernel']}, mse={row['test_mse']:.4f}, time={row['time_total']:.2f}s, mem={row['peak_memory']:.1f}MB")
                    except Exception as e:
                        print(f"Failed: {dataset_name}, {method}, lam={lam}, kernel={kcfg['kernel']}: {e}")

        all_results.extend(results)

    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv("experiment_results.csv", index=False)
    print("Results saved to experiment_results.csv")

    # Save minimal table
    minimal_df = df[['dataset', 'lam', 'kernel', 'method', 'peak_memory', 'time_total', 'pred_time', 'iters', 'test_mse']]
    minimal_df.to_csv("minimal_results.csv", index=False)
    print("Minimal results saved to minimal_results.csv")

    # Plot ratios
    plot_ratios()

def plot_ratios():
    df = pd.read_csv("minimal_results.csv")
    methods = ["full_exact", "cg", "nystrom_pcg", "rpchol_pcg", "restricted_rpchol", "restricted_block_rpchol", "restricted_rls", "restricted_k_means", "krill", "afn_pcg"]

    datasets = df['dataset'].unique()

    for dataset in datasets:
        df_subset = df[df['dataset'] == dataset]

        ratios = {method: {'learn_time': [], 'memory': [], 'pred_time': [], 'mse': []} for method in methods}

        for (lam, kernel), group in df_subset.groupby(['lam', 'kernel']):
            if 'full_exact' not in group['method'].values:
                continue
            full_row = group[group['method'] == 'full_exact'].iloc[0]
            full_learn_time = full_row['time_total'] - full_row['pred_time']
            full_memory = full_row['peak_memory']
            full_pred_time = full_row['pred_time']
            full_mse = full_row['test_mse']

            for _, row in group.iterrows():
                method = row['method']
                if method not in ratios:
                    continue
                learn_time = row['time_total'] - row['pred_time']
                ratios[method]['learn_time'].append(learn_time / full_learn_time)
                ratios[method]['memory'].append(row['peak_memory'] / full_memory)
                ratios[method]['pred_time'].append(row['pred_time'] / full_pred_time)
                ratios[method]['mse'].append(row['test_mse'] / full_mse)

        # Compute averages
        avg_ratios = {method: {} for method in methods}
        for method in methods:
            for metric in ['learn_time', 'memory', 'pred_time', 'mse']:
                if ratios[method][metric]:
                    avg_ratios[method][metric] = np.mean(ratios[method][metric])
                else:
                    avg_ratios[method][metric] = 1.0  # for full_exact

        # Plot horizontal bar plots for each metric with colors by value
        metrics = ['learn_time', 'memory', 'pred_time', 'mse']
        metric_names = ['Learning Time', 'Memory Usage', 'Prediction Time', 'MSE']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        dataset_title = dataset.upper() + " Dataset"
        fig.suptitle(dataset_title, fontsize=16)

        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i]
            y_pos = np.arange(len(methods))
            values = [1 - avg_ratios[method][metric] for method in methods]  # 1 - ratio
            # Colors by value
            colors = [plt.cm.viridis(val) for val in values]
            ax.barh(y_pos, values, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(methods)
            ax.set_xlabel('1 - Ratio to Full Exact')
            ax.set_title(f'{name} (1 - Ratio)')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'ratios_{dataset}_horizontal_bar_plots.png', dpi=300, bbox_inches='tight')
        print(f"Plots saved to ratios_{dataset}_horizontal_bar_plots.png")
        # plt.show()  # Commented out to avoid blocking in script mode

if __name__ == "__main__":
    main()
