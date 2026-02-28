"""
Comparison of theoretical extensions to the inverse seam design QP.

This script tests whether two natural extensions of the inverse seam theory
(Theorem 7) yield better recovery on the Stanford Bunny mesh:

Extension A — Weighted Least Squares (WLS)
------------------------------------------
The noise model is multiplicative: w* = w_gt · (1 + σξ), so
    Var(w*_e) = σ² w_gt(e)².
The current (unweighted) QP treats all edges equally. WLS weights each
residual by 1/w*(e), minimising *relative* rather than *absolute* error:

    min_X  Σ_e [ (α_e (X_u + X_v) - w*_e) / w*_e ]²

This is still quadratic in X (replace α_e → α_e/w*_e, w*_e → 1) and
preserves strict convexity on non-bipartite graphs.

Extension B — Graph-Laplacian smoothness regularisation in X
------------------------------------------------------------
Add a penalty  μ Σ_{(u,v)∈E} (X_u - X_v)² = μ X^T L X,  where L is
the combinatorial graph Laplacian. This penalises rapid variation of the
conformal factor and still yields a QP (H + μ L is PD for μ ≥ 0
on a connected non-bipartite graph).

Both extensions can be combined (WLS + regularisation).
We also test iteratively reweighted LS (IRLS): solve the QP, reweight
using the recovered edge weights, solve again.
"""

from __future__ import annotations
import time
import numpy as np
import scipy.sparse as sp
from scipy.optimize import lsq_linear
from scipy.sparse.linalg import spsolve

# Reuse bunny infrastructure
import bunny


def solve_inverse_seam_extended(
    vertices: np.ndarray,
    edges: np.ndarray,
    w_star: np.ndarray,
    *,
    use_wls: bool = False,
    reg_mu: float = 0.0,
    irls_iters: int = 0,
) -> tuple[np.ndarray, dict]:
    """Extended inverse-design QP with WLS and/or Laplacian regularisation.

    Parameters
    ----------
    vertices : (N, D)
    edges : (E, 2)
    w_star : (E,)  target edge weights
    use_wls : if True, weight residuals by 1/w*
    reg_mu : Laplacian smoothness penalty weight (0 = no penalty)
    irls_iters : number of iteratively-reweighted LS iterations (0 = single solve)

    Returns
    -------
    X_opt : (N,) recovered conformal factors
    info : dict with solver metadata
    """
    N = len(vertices)
    E = len(edges)
    u = edges[:, 0].astype(np.int64)
    v = edges[:, 1].astype(np.int64)
    l0 = np.linalg.norm(vertices[u] - vertices[v], axis=1)
    alpha = l0 / 2.0  # trapezoidal coefficient

    def _build_and_solve(alpha_eff, w_target):
        """Build and solve the QP for given effective alpha and target."""
        alpha_sq = alpha_eff ** 2
        # Hessian (signless Laplacian structure)
        I = np.concatenate([u, v])
        J = np.concatenate([v, u])
        V_off = np.concatenate([alpha_sq, alpha_sq])
        H = sp.coo_matrix((V_off, (I, J)), shape=(N, N))
        H_diag = np.bincount(u, weights=alpha_sq, minlength=N) + \
                 np.bincount(v, weights=alpha_sq, minlength=N)
        H = (H + sp.diags(H_diag)).tocsr()

        # RHS
        b_edge = alpha_eff * w_target
        b = np.bincount(u, weights=b_edge, minlength=N) + \
            np.bincount(v, weights=b_edge, minlength=N)

        # Add Laplacian regularisation if requested
        if reg_mu > 0:
            # Combinatorial graph Laplacian: L = D - A
            ones_E = np.ones(E)
            A_adj = sp.coo_matrix(
                (np.concatenate([ones_E, ones_E]),
                 (np.concatenate([u, v]), np.concatenate([v, u]))),
                shape=(N, N),
            )
            deg = np.bincount(u, minlength=N) + np.bincount(v, minlength=N)
            L = sp.diags(deg.astype(float)) - A_adj.tocsr()
            H = H + reg_mu * L.tocsr()

        # Small ridge for numerical stability
        ridge = 1e-12 * max(1.0, float(H.diagonal().mean()))
        H = H + sp.eye(N, format="csr") * ridge

        X = spsolve(H, b)

        # Positivity check — use sparse lsq_linear fallback (scales to large N)
        if np.any(~np.isfinite(X)) or np.min(X) < -1e-8 * max(1e-12, np.median(np.abs(X))):
            # Build sparse A matrix: A[e, u_e] = alpha_eff_e, A[e, v_e] = alpha_eff_e
            rows_A = np.concatenate([np.arange(E, dtype=np.int64),
                                     np.arange(E, dtype=np.int64)])
            cols_A = np.concatenate([u, v])
            data_A = np.concatenate([alpha_eff, alpha_eff])
            n_rows = E

            if reg_mu > 0:
                # Augment with sqrt(mu)*(X_u - X_v) rows for smoothness
                sqrt_mu = np.sqrt(reg_mu)
                reg_rows = np.concatenate([np.arange(E, dtype=np.int64),
                                           np.arange(E, dtype=np.int64)])
                reg_cols = np.concatenate([u, v])
                reg_data = np.concatenate([np.full(E, sqrt_mu),
                                           np.full(E, -sqrt_mu)])
                rows_A = np.concatenate([rows_A, reg_rows + E])
                cols_A = np.concatenate([cols_A, reg_cols])
                data_A = np.concatenate([data_A, reg_data])
                n_rows = 2 * E

            A_sp = sp.coo_matrix((data_A, (rows_A, cols_A)),
                                 shape=(n_rows, N)).tocsr()
            rhs = np.concatenate([w_target, np.zeros(E)]) if reg_mu > 0 else w_target
            res = lsq_linear(A_sp, rhs, bounds=(0.0, np.inf),
                             method="trf", max_iter=200)
            X = res.x

        X = np.clip(X, 1e-12, None)
        return X

    # Initial WLS weights
    if use_wls:
        wls_weights = 1.0 / np.maximum(w_star, 1e-12)
        alpha_eff = alpha * wls_weights
        w_target = w_star * wls_weights  # = ones
    else:
        alpha_eff = alpha
        w_target = w_star

    X_opt = _build_and_solve(alpha_eff, w_target)

    # IRLS iterations
    for it in range(irls_iters):
        w_current = l0 * (X_opt[u] + X_opt[v]) / 2.0
        iter_weights = 1.0 / np.maximum(w_current, 1e-12)
        alpha_eff_it = alpha * iter_weights
        w_target_it = w_star * iter_weights
        X_opt = _build_and_solve(alpha_eff_it, w_target_it)

    info = {
        "use_wls": use_wls,
        "reg_mu": reg_mu,
        "irls_iters": irls_iters,
    }
    return X_opt, info


def run_comparison(sigma: float = 0.10, seed: int = 42):
    """Run all solver variants on the Stanford Bunny and compare."""
    t0 = time.perf_counter()

    # Load mesh (same as bunny.py)
    obj_path = bunny.download_bunny()
    vertices, faces = bunny.load_obj(obj_path)
    edges = bunny.extract_edges(faces)
    vertices, faces, edges = bunny.largest_component(vertices, faces, edges)
    N, E, F = len(vertices), len(edges), len(faces)
    print(f"Mesh: {N:,} verts, {E:,} edges, {F:,} faces\n")

    # Ground truth
    s_gt = bunny.seam_gt_3d(vertices)
    X_gt = np.exp(s_gt)
    u, v = edges[:, 0], edges[:, 1]
    l0 = np.linalg.norm(vertices[u] - vertices[v], axis=1)
    w_gt = l0 * (X_gt[u] + X_gt[v]) / 2.0

    # Noisy targets
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal(E)
    w_star = np.clip(w_gt * (1.0 + sigma * xi), 1e-6, None)

    noisy_rel = float(np.linalg.norm(w_star - w_gt) / np.linalg.norm(w_gt))

    import validation

    # Define solver variants to compare
    variants = [
        ("Baseline (unweighted LS)", dict(use_wls=False, reg_mu=0.0, irls_iters=0)),
        ("WLS (relative error)", dict(use_wls=True, reg_mu=0.0, irls_iters=0)),
        ("LS + Laplacian (μ=0.001)", dict(use_wls=False, reg_mu=0.001, irls_iters=0)),
        ("LS + Laplacian (μ=0.01)", dict(use_wls=False, reg_mu=0.01, irls_iters=0)),
        ("LS + Laplacian (μ=0.1)", dict(use_wls=False, reg_mu=0.1, irls_iters=0)),
        ("WLS + Laplacian (μ=0.001)", dict(use_wls=True, reg_mu=0.001, irls_iters=0)),
        ("WLS + Laplacian (μ=0.01)", dict(use_wls=True, reg_mu=0.01, irls_iters=0)),
        ("WLS + Laplacian (μ=0.1)", dict(use_wls=True, reg_mu=0.1, irls_iters=0)),
        ("IRLS (2 iter)", dict(use_wls=True, reg_mu=0.0, irls_iters=2)),
        ("IRLS (5 iter)", dict(use_wls=True, reg_mu=0.0, irls_iters=5)),
        ("IRLS + Lap (2 iter, μ=0.01)", dict(use_wls=True, reg_mu=0.01, irls_iters=2)),
    ]

    print(f"{'Variant':<36s} {'w_err':>8s} {'improv':>8s} {'Pearson r':>10s} {'R²':>8s} {'time':>7s}")
    print("=" * 84)

    for name, kwargs in variants:
        t1 = time.perf_counter()
        X_opt, info = solve_inverse_seam_extended(vertices, edges, w_star, **kwargs)
        dt = time.perf_counter() - t1

        s_opt = np.log(X_opt)
        w_opt = l0 * (X_opt[u] + X_opt[v]) / 2.0

        weight_err = float(np.linalg.norm(w_opt - w_gt) / np.linalg.norm(w_gt))
        improvement = noisy_rel / (weight_err + 1e-15)
        pearson_r = float(np.corrcoef(s_gt, s_opt)[0, 1])
        a, b, r2 = validation.compute_affine_fit(s_gt, s_opt)

        print(
            f"{name:<36s} {weight_err:8.6f} {improvement:7.1f}× "
            f"{pearson_r:10.6f} {r2:8.6f} {dt:6.2f}s"
        )

    print(f"\nNoisy edge-weight error:  {noisy_rel:.6f}")
    print(f"Total time: {time.perf_counter() - t0:.1f}s")


def run_sigma_sweep(sigmas=None, seed: int = 42):
    """Run comparison across multiple noise levels."""
    if sigmas is None:
        sigmas = [0.05, 0.10, 0.15, 0.20]

    obj_path = bunny.download_bunny()
    vertices, faces = bunny.load_obj(obj_path)
    edges = bunny.extract_edges(faces)
    vertices, faces, edges = bunny.largest_component(vertices, faces, edges)
    N, E, F = len(vertices), len(edges), len(faces)

    s_gt = bunny.seam_gt_3d(vertices)
    X_gt = np.exp(s_gt)
    u, v = edges[:, 0], edges[:, 1]
    l0 = np.linalg.norm(vertices[u] - vertices[v], axis=1)
    w_gt = l0 * (X_gt[u] + X_gt[v]) / 2.0

    import validation

    # Focused comparison: baseline vs best candidates
    methods = [
        ("Baseline", dict(use_wls=False, reg_mu=0.0, irls_iters=0)),
        ("WLS", dict(use_wls=True, reg_mu=0.0, irls_iters=0)),
        ("WLS+Lap(0.01)", dict(use_wls=True, reg_mu=0.01, irls_iters=0)),
        ("IRLS(2)+Lap(0.01)", dict(use_wls=True, reg_mu=0.01, irls_iters=2)),
    ]

    print(f"\n{'σ':>5s}  {'Method':<24s} {'w_err':>8s} {'×improv':>8s} {'Pearson r':>10s} {'R²':>8s}")
    print("=" * 72)

    for sigma in sigmas:
        rng = np.random.default_rng(seed)
        xi = rng.standard_normal(E)
        w_star = np.clip(w_gt * (1.0 + sigma * xi), 1e-6, None)
        noisy_rel = float(np.linalg.norm(w_star - w_gt) / np.linalg.norm(w_gt))

        for name, kwargs in methods:
            X_opt, _ = solve_inverse_seam_extended(vertices, edges, w_star, **kwargs)
            s_opt = np.log(X_opt)
            w_opt = l0 * (X_opt[u] + X_opt[v]) / 2.0

            weight_err = float(np.linalg.norm(w_opt - w_gt) / np.linalg.norm(w_gt))
            improvement = noisy_rel / (weight_err + 1e-15)
            pearson_r = float(np.corrcoef(s_gt, s_opt)[0, 1])
            _, _, r2 = validation.compute_affine_fit(s_gt, s_opt)

            print(
                f"{sigma:5.2f}  {name:<24s} {weight_err:8.6f} {improvement:7.1f}× "
                f"{pearson_r:10.6f} {r2:8.6f}"
            )
        print()


if __name__ == "__main__":
    print("=" * 72)
    print("  Extension Comparison — Stanford Bunny (σ=0.10)")
    print("=" * 72)
    run_comparison(sigma=0.10, seed=42)

    print("\n\n")
    print("=" * 72)
    print("  Cross-σ comparison of best extensions")
    print("=" * 72)
    run_sigma_sweep()
