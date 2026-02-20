"""
Seam-Driven Geometry: Synthetic Numerical Validation
This script implements the "Metric Nearness Projection" via the Inverse Seam Theorem 
(Theorem 7) from the paper "Discrete-to-Continuum Metrics from Scalar Fields".

It demonstrates that recovering a metric space from noisy, physically invalid target 
edge weights reduces to solving a strictly convex Quadratic Program governed by the 
Signless Laplacian of the mesh.

Dependencies: numpy, scipy, matplotlib
"""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import lsq_linear
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


RUN_SWEEP = True
PLOT_EXAMPLE = True

SAMPLE_SOURCES = 12
SAMPLE_TARGETS_PER_SOURCE = 12


def _objective_and_derivatives(
    N: int,
    u: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    w_star: np.ndarray,
    X: np.ndarray,
):
    """Objective and its derivatives for

        f(X) = 1/2 * sum_e ( alpha_e * (X_u + X_v) - w*_e )^2

    Returns:
        f: scalar
        g: (N,) gradient
        H: (N,N) sparse Hessian (constant in X)
    """
    E = len(u)
    if len(v) != E or len(alpha) != E or len(w_star) != E:
        raise ValueError("Edge arrays must have consistent length")

    r = alpha * (X[u] + X[v]) - w_star
    f = 0.5 * float(np.dot(r, r))

    # g = A^T r where A[e,u]=alpha_e, A[e,v]=alpha_e
    g = np.bincount(u, weights=alpha * r, minlength=N) + np.bincount(v, weights=alpha * r, minlength=N)

    # H = A^T A
    alpha_sq = alpha**2
    I = np.concatenate([u, v])
    J = np.concatenate([v, u])
    V_off = np.concatenate([alpha_sq, alpha_sq])
    H_off = sp.coo_matrix((V_off, (I, J)), shape=(N, N))
    H_diag_vals = np.bincount(u, weights=alpha_sq, minlength=N) + np.bincount(v, weights=alpha_sq, minlength=N)
    H = (H_off + sp.diags(H_diag_vals)).tocsr()

    return f, g, H


def derivative_check(points: np.ndarray, edges: np.ndarray, w_star: np.ndarray, seed: int = 123):
    """Finite-difference check of gradient (Jacobian) and Hessian formulas.

    This validates that the analytic gradient matches finite differences of f,
    and that the Hessian action matches finite differences of the gradient.
    """
    rng = np.random.default_rng(seed)
    N = len(points)
    u = edges[:, 0].astype(np.int64)
    v = edges[:, 1].astype(np.int64)
    l0 = np.linalg.norm(points[u] - points[v], axis=1)
    alpha = l0 / 2.0

    # Use a strictly positive random test point.
    X0 = np.exp(0.1 * rng.standard_normal(N))
    p = rng.standard_normal(N)
    p /= (np.linalg.norm(p) + 1e-12)

    f0, g0, H = _objective_and_derivatives(N, u, v, alpha, w_star, X0)
    Hp = H @ p

    # Directional derivative check: (f(x+eps p) - f(x-eps p)) / (2eps) ≈ g(x)·p
    eps = 1e-6
    f_plus, _, _ = _objective_and_derivatives(N, u, v, alpha, w_star, X0 + eps * p)
    f_minus, _, _ = _objective_and_derivatives(N, u, v, alpha, w_star, X0 - eps * p)
    fd_dir = (f_plus - f_minus) / (2.0 * eps)
    an_dir = float(np.dot(g0, p))
    dir_abs_err = float(abs(fd_dir - an_dir))
    dir_rel_err = float(dir_abs_err / (abs(fd_dir) + abs(an_dir) + 1e-12))

    # Hessian-vector check via gradient difference: (g(x+eps p)-g(x-eps p))/(2eps) ≈ H p
    _, g_plus, _ = _objective_and_derivatives(N, u, v, alpha, w_star, X0 + eps * p)
    _, g_minus, _ = _objective_and_derivatives(N, u, v, alpha, w_star, X0 - eps * p)
    fd_Hp = (g_plus - g_minus) / (2.0 * eps)
    hv_abs_err = float(np.linalg.norm(fd_Hp - Hp))
    hv_rel_err = float(hv_abs_err / (np.linalg.norm(fd_Hp) + np.linalg.norm(Hp) + 1e-12))

    return {
        "f0": float(f0),
        "dir_abs_err": dir_abs_err,
        "dir_rel_err": dir_rel_err,
        "hv_abs_err": hv_abs_err,
        "hv_rel_err": hv_rel_err,
    }

def generate_mesh(n_points: int, rng: np.random.Generator):
    """Generates a random 2D triangulated mesh (non-bipartite graph)."""
    # Generate random points in the unit square
    points = rng.random((n_points, 2))
    tri = Delaunay(points)
    
    # Extract unique edges from the triangulation
    edges = set()
    for simplex in tri.simplices:
        a, b, c = (int(simplex[0]), int(simplex[1]), int(simplex[2]))
        edges.add((min(a, b), max(a, b)))
        edges.add((min(b, c), max(b, c)))
        edges.add((min(c, a), max(c, a)))
    
    edges = np.array(list(edges))
    return points, edges, tri.simplices


def solve_inverse_seam(points: np.ndarray, edges: np.ndarray, w_star: np.ndarray):
    """Solve the inverse-design quadratic (unconstrained, with X>=0 fallback).

    Returns:
        X_opt: (N,) recovered X=e^s
        l0: (E,) background edge lengths
        u, v: (E,) edge endpoints
    """
    N = len(points)
    u = edges[:, 0].astype(np.int64)
    v = edges[:, 1].astype(np.int64)

    pts_u, pts_v = points[u], points[v]
    l0 = np.linalg.norm(pts_u - pts_v, axis=1)

    alpha = l0 / 2.0
    alpha_sq = alpha**2

    I = np.concatenate([u, v])
    J = np.concatenate([v, u])
    V_off = np.concatenate([alpha_sq, alpha_sq])

    H_off = sp.coo_matrix((V_off, (I, J)), shape=(N, N))
    H_diag_vals = np.bincount(u, weights=alpha_sq, minlength=N) + np.bincount(v, weights=alpha_sq, minlength=N)
    H = (H_off + sp.diags(H_diag_vals)).tocsr()

    b_edge = alpha * w_star
    b = np.bincount(u, weights=b_edge, minlength=N) + np.bincount(v, weights=b_edge, minlength=N)

    X_opt = spsolve(H, b)

    if not np.all(np.isfinite(X_opt)) or np.min(X_opt) <= 0:
        rows = np.repeat(np.arange(len(edges), dtype=np.int64), 2)
        cols = np.concatenate([u, v])
        data = np.concatenate([alpha, alpha])
        A = sp.coo_matrix((data, (rows, cols)), shape=(len(edges), N)).tocsr()
        res = lsq_linear(A, w_star, bounds=(0.0, np.inf), method="trf")
        if not res.success:
            raise RuntimeError(f"lsq_linear failed: {res.message}")
        X_opt = res.x

    X_opt = np.clip(X_opt, 1e-12, None)
    return X_opt, l0, u, v


def compute_affine_fit(x: np.ndarray, y: np.ndarray):
    """Best-fit affine map y ≈ a*x + b and R^2 (predicting y from x)."""
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    denom = float(np.dot(x_centered, x_centered))
    if denom > 0:
        a = float(np.dot(x_centered, y - y_mean) / denom)
    else:
        a = float("nan")
    b = y_mean - a * x_mean
    residual = y - (a * x + b)
    ss_res = float(np.dot(residual, residual))
    ss_tot = float(np.dot(y - y_mean, y - y_mean))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return a, b, r2


def run_experiment(n_points: int, sigma: float, seed: int = 42):
    rng_points = np.random.default_rng(seed)
    rng_noise = np.random.default_rng(seed + 1)
    rng_pairs = np.random.default_rng(seed + 2)

    points, edges, _faces = generate_mesh(n_points=n_points, rng=rng_points)
    N = len(points)
    E = len(edges)

    u = edges[:, 0].astype(np.int64)
    v = edges[:, 1].astype(np.int64)
    l0 = np.linalg.norm(points[u] - points[v], axis=1)

    s_gt = np.sin(2 * np.pi * points[:, 0]) * np.cos(2 * np.pi * points[:, 1])
    X_gt = np.exp(s_gt)
    w_gt = l0 * (X_gt[u] + X_gt[v]) / 2.0

    # Use a fixed noise direction xi and scale it by sigma for fair sweeps.
    xi = rng_noise.standard_normal(E)
    w_star = w_gt * (1.0 + sigma * xi)
    w_star = np.clip(w_star, 1e-6, None)

    X_opt, l0_check, u_check, v_check = solve_inverse_seam(points, edges, w_star)
    # Sanity: reuse u/v/l0 computed inside solver for weight evaluation
    l0 = l0_check
    u = u_check
    v = v_check
    s_opt = np.log(X_opt)
    w_opt = l0 * (X_opt[u] + X_opt[v]) / 2.0

    # Shortest-path distortion statistics between recovered metric and ground truth.
    # We sample a small set of (source, target) pairs for speed.
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])

    A_gt = sp.coo_matrix((np.concatenate([w_gt, w_gt]), (rows, cols)), shape=(N, N)).tocsr()
    A_star = sp.coo_matrix((np.concatenate([w_star, w_star]), (rows, cols)), shape=(N, N)).tocsr()
    A_opt = sp.coo_matrix((np.concatenate([w_opt, w_opt]), (rows, cols)), shape=(N, N)).tocsr()

    n_sources = min(SAMPLE_SOURCES, N)
    sources = rng_pairs.choice(N, size=n_sources, replace=False)
    # Compute distance vectors for each source under both metrics.
    D_gt = dijkstra(A_gt, directed=False, indices=sources)
    D_star = dijkstra(A_star, directed=False, indices=sources)
    D_opt = dijkstra(A_opt, directed=False, indices=sources)

    rel_errors_noisy = []
    rel_errors_opt = []
    distortions_noisy = []
    distortions_opt = []
    for i, src in enumerate(sources):
        # Avoid trivial targets (src itself) and unreachable nodes (shouldn't happen for Delaunay graphs).
        candidates = np.arange(N)
        candidates = candidates[candidates != src]
        n_t = min(SAMPLE_TARGETS_PER_SOURCE, len(candidates))
        targets = rng_pairs.choice(candidates, size=n_t, replace=False)

        dgt = D_gt[i, targets]
        dstar = D_star[i, targets]
        dopt = D_opt[i, targets]
        mask = np.isfinite(dgt) & np.isfinite(dstar) & np.isfinite(dopt) & (dgt > 0)
        if not np.any(mask):
            continue
        dgt = dgt[mask]
        dstar = dstar[mask]
        dopt = dopt[mask]

        rel_noisy = np.abs(dstar - dgt) / dgt
        rel_opt = np.abs(dopt - dgt) / dgt
        rel_errors_noisy.append(rel_noisy)
        rel_errors_opt.append(rel_opt)
        distortions_noisy.append(dstar / dgt)
        distortions_opt.append(dopt / dgt)

    if rel_errors_opt:
        rel_noisy_all = np.concatenate(rel_errors_noisy)
        rel_opt_all = np.concatenate(rel_errors_opt)

        dist_noisy_all = np.concatenate(distortions_noisy)
        dist_opt_all = np.concatenate(distortions_opt)

        path_rel_mean_noisy = float(np.mean(rel_noisy_all))
        path_rel_p95_noisy = float(np.percentile(rel_noisy_all, 95))
        dist_ratio_mean_noisy = float(np.mean(dist_noisy_all))

        path_rel_mean = float(np.mean(rel_opt_all))
        path_rel_p95 = float(np.percentile(rel_opt_all, 95))
        path_rel_max = float(np.max(rel_opt_all))
        dist_ratio_mean = float(np.mean(dist_opt_all))
    else:
        path_rel_mean_noisy = float("nan")
        path_rel_p95_noisy = float("nan")
        dist_ratio_mean_noisy = float("nan")
        path_rel_mean = float("nan")
        path_rel_p95 = float("nan")
        path_rel_max = float("nan")
        dist_ratio_mean = float("nan")

    path_rel_mean_improve = float(path_rel_mean_noisy - path_rel_mean) if np.isfinite(path_rel_mean_noisy) and np.isfinite(path_rel_mean) else float("nan")
    path_rel_p95_improve = float(path_rel_p95_noisy - path_rel_p95) if np.isfinite(path_rel_p95_noisy) and np.isfinite(path_rel_p95) else float("nan")

    metrics = {
        "n_points": int(N),
        "n_edges": int(E),
        "sigma": float(sigma),
        "mean_l0": float(np.mean(l0)),
        "max_l0": float(np.max(l0)),
        "fit_to_target": float(np.linalg.norm(w_opt - w_star) / np.linalg.norm(w_star)),
        "noisy_to_truth": float(np.linalg.norm(w_star - w_gt) / np.linalg.norm(w_gt)),
        "denoised_to_truth": float(np.linalg.norm(w_opt - w_gt) / np.linalg.norm(w_gt)),
        "X_rel_err": float(np.linalg.norm(X_opt - X_gt) / np.linalg.norm(X_gt)),
        "s_rel_err": float(np.linalg.norm(s_opt - s_gt) / np.linalg.norm(s_gt)),
        "pearson_r": float(np.corrcoef(s_gt, s_opt)[0, 1]),
        "path_rel_mean_noisy": path_rel_mean_noisy,
        "path_rel_p95_noisy": path_rel_p95_noisy,
        "dist_ratio_mean_noisy": dist_ratio_mean_noisy,
        "path_rel_mean": path_rel_mean,
        "path_rel_p95": path_rel_p95,
        "path_rel_mean_improve": path_rel_mean_improve,
        "path_rel_p95_improve": path_rel_p95_improve,
        "path_rel_max": path_rel_max,
        "dist_ratio_mean": dist_ratio_mean,
    }
    a_fit, b_fit, r2_fit = compute_affine_fit(s_gt, s_opt)
    metrics.update({"a_fit": float(a_fit), "b_fit": float(b_fit), "r2_fit": float(r2_fit)})
    return points, edges, s_gt, s_opt, w_gt, w_star, w_opt, metrics

def main():
    if RUN_SWEEP:
        n_grid = [200, 400, 800, 1200]
        sigma_grid = [0.00, 0.05, 0.10, 0.15, 0.20]

        print("--- Seam-Driven Geometry Validation (sweep) ---")
        print(
            "Columns: n  E  sigma  mean_l0  max_l0  fit(w_opt,w*)  noisy->truth  denoise->truth  "
            "path_rel_mean_noisy  path_rel_p95_noisy  path_rel_mean  path_rel_p95  "
            "d_path_mean  d_path_p95  corr  a_fit  b_fit  R2"
        )
        results = []
        deriv_checks = []

        example_payload = None
        example_key = (800, 0.15)

        for n in n_grid:
            for sigma in sigma_grid:
                points, edges, s_gt, s_opt, w_gt, w_star, w_opt, m = run_experiment(n_points=n, sigma=sigma, seed=42)
                results.append(m)
                chk = derivative_check(points, edges, w_star, seed=123)
                deriv_checks.append({"n": n, "sigma": float(sigma), **chk})
                print(
                    f"{m['n_points']:4d} {m['n_edges']:5d} {m['sigma']:5.2f} "
                    f"{m['mean_l0']:.4f} {m['max_l0']:.4f} "
                    f"{m['fit_to_target']:.4f} {m['noisy_to_truth']:.4f} {m['denoised_to_truth']:.4f} "
                    f"{m['path_rel_mean_noisy']:.4f} {m['path_rel_p95_noisy']:.4f} "
                    f"{m['path_rel_mean']:.4f} {m['path_rel_p95']:.4f} "
                    f"{m['path_rel_mean_improve']:.4f} {m['path_rel_p95_improve']:.4f} "
                    f"{m['pearson_r']:.4f} {m['a_fit']:.4f} {m['b_fit']:.4f} {m['r2_fit']:.4f}"
                )
                if (n, float(sigma)) == example_key:
                    example_payload = (points, edges, s_gt, s_opt, m)

        if deriv_checks:
            worst_dir = max(deriv_checks, key=lambda d: d.get("dir_rel_err", float("-inf")))
            worst_hv = max(deriv_checks, key=lambda d: d.get("hv_rel_err", float("-inf")))
            print("\n--- Derivative Check (finite differences; worst over sweep) ---")
            print(
                "Directional deriv rel err : "
                f"{worst_dir['dir_rel_err']:.3e} (abs {worst_dir['dir_abs_err']:.3e}) "
                f"at (n={worst_dir['n']}, sigma={worst_dir['sigma']:.2f})"
            )
            print(
                "Hessian-vector rel err    : "
                f"{worst_hv['hv_rel_err']:.3e} (abs {worst_hv['hv_abs_err']:.3e}) "
                f"at (n={worst_hv['n']}, sigma={worst_hv['sigma']:.2f})"
            )

        # Optional single representative plot
        if PLOT_EXAMPLE and example_payload is not None:
            points, edges, s_gt, s_opt, m = example_payload
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            sc1 = axes[0].scatter(points[:, 0], points[:, 1], c=s_gt, cmap='viridis', s=20)
            axes[0].set_title("Ground Truth Seam ($s_{gt}$)")
            axes[0].axis('equal'); axes[0].axis('off')
            fig.colorbar(sc1, ax=axes[0], fraction=0.046, pad=0.04)

            sc2 = axes[1].scatter(points[:, 0], points[:, 1], c=s_opt, cmap='viridis', s=20)
            axes[1].set_title("Recovered Seam ($s^*$) from Noisy Metric")
            axes[1].axis('equal'); axes[1].axis('off')
            fig.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.04)

            axes[2].scatter(s_gt, s_opt, alpha=0.5, s=10, c='black')
            lo = float(min(np.min(s_gt), np.min(s_opt)))
            hi = float(max(np.max(s_gt), np.max(s_opt)))
            axes[2].plot([lo, hi], [lo, hi], 'r--', lw=2)
            axes[2].plot([lo, hi], [m['a_fit'] * lo + m['b_fit'], m['a_fit'] * hi + m['b_fit']], color='blue', lw=2)
            axes[2].set_title(f"Seam Recovery Correlation (n={m['n_points']}, sigma={m['sigma']:.2f})")
            axes[2].set_xlabel("Ground Truth Seam Value")
            axes[2].set_ylabel("Recovered Seam Value")
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            print("\nRendering example visualization...")
            plt.show()
    else:
        # Single-run mode (kept for convenience)
        points, edges, s_gt, s_opt, _w_gt, w_star, _w_opt, m = run_experiment(n_points=800, sigma=0.15, seed=42)
        print("--- Seam-Driven Geometry Validation (single) ---")
        print(f"Mesh generated: {m['n_points']} vertices, {m['n_edges']} edges.")
        print(f"Added {m['sigma']*100:.1f}% Gaussian noise to ground-truth metric.")
        print("\n--- Results ---")
        print(f"Target fitting error (||w_opt - w*||_2 / ||w*||_2) : {m['fit_to_target']:.4f}")
        print(f"Noisy-to-truth weight error (||w* - w_gt||_2 / ||w_gt||_2): {m['noisy_to_truth']:.4f}")
        print(f"Denoised-to-truth weight error (||w_opt - w_gt||_2 / ||w_gt||_2): {m['denoised_to_truth']:.4f}")
        print(f"Path rel error noisy->truth (mean / p95): {m['path_rel_mean_noisy']:.4f} / {m['path_rel_p95_noisy']:.4f}")
        print(f"Path rel error denoised->truth (mean / p95): {m['path_rel_mean']:.4f} / {m['path_rel_p95']:.4f}")
        print(f"Path rel improvement (mean / p95): {m['path_rel_mean_improve']:.4f} / {m['path_rel_p95_improve']:.4f}")
        print(f"Pearson corr(s_gt, s_opt) : {m['pearson_r']:.4f}")
        print(f"Affine fit: s_opt ≈ a*s_gt + b (a, b) : ({m['a_fit']:.4f}, {m['b_fit']:.4f})")
        print(f"Affine fit R^2 : {m['r2_fit']:.4f}")

        chk = derivative_check(points, edges, w_star, seed=123)
        print("\n--- Derivative Check (finite differences) ---")
        print(f"Directional deriv rel err : {chk['dir_rel_err']:.3e} (abs {chk['dir_abs_err']:.3e})")
        print(f"Hessian-vector rel err    : {chk['hv_rel_err']:.3e} (abs {chk['hv_abs_err']:.3e})")

        if PLOT_EXAMPLE:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            sc1 = axes[0].scatter(points[:, 0], points[:, 1], c=s_gt, cmap='viridis', s=20)
            axes[0].set_title("Ground Truth Seam ($s_{gt}$)")
            axes[0].axis('equal'); axes[0].axis('off')
            fig.colorbar(sc1, ax=axes[0], fraction=0.046, pad=0.04)

            sc2 = axes[1].scatter(points[:, 0], points[:, 1], c=s_opt, cmap='viridis', s=20)
            axes[1].set_title("Recovered Seam ($s^*$) from Noisy Metric")
            axes[1].axis('equal'); axes[1].axis('off')
            fig.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.04)

            axes[2].scatter(s_gt, s_opt, alpha=0.5, s=10, c='black')
            lo = float(min(np.min(s_gt), np.min(s_opt)))
            hi = float(max(np.max(s_gt), np.max(s_opt)))
            axes[2].plot([lo, hi], [lo, hi], 'r--', lw=2)
            axes[2].plot([lo, hi], [m['a_fit'] * lo + m['b_fit'], m['a_fit'] * hi + m['b_fit']], color='blue', lw=2)
            axes[2].set_title("Seam Recovery Correlation")
            axes[2].set_xlabel("Ground Truth Seam Value")
            axes[2].set_ylabel("Recovered Seam Value")
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            print("\nRendering visualization...")
            plt.show()

if __name__ == "__main__":
    main()