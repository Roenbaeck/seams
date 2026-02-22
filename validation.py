"""
Seam-Driven Geometry: Synthetic Numerical Validation
This script implements the "Metric Nearness Projection" via the Inverse Seam Theorem 
(Theorem 7) from the paper "Discrete-to-Continuum Metrics from Scalar Fields".

It demonstrates that recovering a metric space from noisy, physically invalid target 
edge weights reduces to solving a strictly convex Quadratic Program governed by the 
Signless Laplacian of the mesh.

Dependencies: numpy, scipy, matplotlib
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import lsq_linear
from scipy.optimize import nnls
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import Delaunay, cKDTree
import matplotlib.pyplot as plt

SAMPLE_SOURCES = 50
SAMPLE_TARGETS_PER_SOURCE = 50

# The convergence study can take additional time (extra meshes + a dense grid Dijkstra).
RUN_CONVERGENCE_STUDY = True

# Mesh construction settings.
# Delaunay triangulation is great for quality meshes (and gives faces), but can become
# slow and occasionally fragile at large N due to QHull behavior and memory.
DELAUNAY_MAX_N = 5000
KNN_K = 8


def seam_gt(points: np.ndarray) -> np.ndarray:
    """Ground-truth seam field used throughout the validation."""
    return np.sin(2 * np.pi * points[:, 0]) * np.cos(2 * np.pi * points[:, 1])


def _angles_from_edge_lengths(a: float, b: float, c: float):
    """Return internal angles (A,B,C) for side lengths (a,b,c).

    Angles are opposite corresponding side lengths.
    """

    def safe_acos(x: float) -> float:
        return float(np.arccos(np.clip(x, -1.0, 1.0)))

    eps = 1e-15
    a = float(max(a, eps))
    b = float(max(b, eps))
    c = float(max(c, eps))

    # Law of cosines
    cosA = (b * b + c * c - a * a) / (2.0 * b * c)
    cosB = (a * a + c * c - b * b) / (2.0 * a * c)
    cosC = (a * a + b * b - c * c) / (2.0 * a * b)
    return safe_acos(cosA), safe_acos(cosB), safe_acos(cosC)


def compute_discrete_gaussian_curvature(
    n_vertices: int,
    faces: np.ndarray,
    boundary_vertices: np.ndarray,
    edge_length_map: dict[tuple[int, int], float],
):
    """Compute angle-defect discrete Gaussian curvature at vertices.

    For interior vertices u: K(u) = 2*pi - sum_t theta_t(u)
    For boundary vertices u: K(u) =   pi - sum_t theta_t(u)
    """
    if faces is None:
        raise ValueError("faces is required to compute curvature")

    angle_sum = np.zeros(int(n_vertices), dtype=float)
    faces = np.asarray(faces, dtype=np.int64)

    for (i, j, k) in faces:
        ij = edge_length_map.get((min(i, j), max(i, j)))
        jk = edge_length_map.get((min(j, k), max(j, k)))
        ki = edge_length_map.get((min(k, i), max(k, i)))
        if ij is None or jk is None or ki is None:
            # Should not happen for Delaunay-derived edge sets, but keep robust.
            continue

        # Side lengths opposite vertices (i,j,k)
        a = float(jk)  # opposite i
        b = float(ki)  # opposite j
        c = float(ij)  # opposite k
        Ai, Aj, Ak = _angles_from_edge_lengths(a, b, c)
        angle_sum[i] += Ai
        angle_sum[j] += Aj
        angle_sum[k] += Ak

    boundary_mask = np.zeros(int(n_vertices), dtype=bool)
    if boundary_vertices is not None and len(boundary_vertices) > 0:
        boundary_mask[np.asarray(boundary_vertices, dtype=np.int64)] = True

    K = np.empty(int(n_vertices), dtype=float)
    K[boundary_mask] = np.pi - angle_sum[boundary_mask]
    K[~boundary_mask] = 2.0 * np.pi - angle_sum[~boundary_mask]
    return K, angle_sum


def curvature_relative_error(K_gt: np.ndarray, K_est: np.ndarray):
    """Relative error summary between two curvature vectors."""
    K_gt = np.asarray(K_gt, dtype=float)
    K_est = np.asarray(K_est, dtype=float)
    eps = 1e-12
    rel_l2 = float(np.linalg.norm(K_est - K_gt) / (np.linalg.norm(K_gt) + eps))
    abs_rel = np.abs(K_est - K_gt) / (np.abs(K_gt) + eps)
    return {
        "curv_rel_l2": rel_l2,
        "curv_abs_rel_mean": float(np.mean(abs_rel)),
        "curv_abs_rel_p95": float(np.percentile(abs_rel, 95)),
        "curv_abs_rel_max": float(np.max(abs_rel)),
    }


def plot_sweep_summary(results):
    """Plot key sweep metrics as curves vs sigma for each mesh size.

    This complements the single-example plots by showing global trends.
    """
    if not results:
        return

    n_values = sorted({int(m["n_points"]) for m in results})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(n_values), 1)))

    # Panel 1: edge-weight error denoising
    ax = axes[0]
    for color, n in zip(colors, n_values):
        rows = sorted([m for m in results if int(m["n_points"]) == n], key=lambda r: float(r["sigma"]))
        sig = np.array([float(r["sigma"]) for r in rows])
        noisy = np.array([float(r["noisy_to_truth"]) for r in rows])
        den = np.array([float(r["denoised_to_truth"]) for r in rows])
        ax.plot(sig, den, color=color, marker="o", label=f"n={n}")
        ax.plot(sig, noisy, color=color, linestyle="--", marker="x", alpha=0.6)
    ax.set_title("Edge-weight error vs noise (solid=denoised, dashed=noisy)")
    ax.set_xlabel("sigma")
    ax.set_ylabel("relative error")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Mesh size", fontsize=9)

    # Panel 2: shortest-path distortion
    ax = axes[1]
    for color, n in zip(colors, n_values):
        rows = sorted([m for m in results if int(m["n_points"]) == n], key=lambda r: float(r["sigma"]))
        sig = np.array([float(r["sigma"]) for r in rows])
        noisy = np.array([float(r["path_rel_mean_noisy"]) for r in rows])
        den = np.array([float(r["path_rel_mean"]) for r in rows])
        ax.plot(sig, den, color=color, marker="o")
        ax.plot(sig, noisy, color=color, linestyle="--", marker="x", alpha=0.6)
    ax.set_title("Mean shortest-path distortion (solid=denoised, dashed=noisy)")
    ax.set_xlabel("sigma")
    ax.set_ylabel("mean |d̂ - d| / d")
    ax.grid(True, alpha=0.3)

    # Panel 3: seam recovery correlation
    ax = axes[2]
    for color, n in zip(colors, n_values):
        rows = sorted([m for m in results if int(m["n_points"]) == n], key=lambda r: float(r["sigma"]))
        sig = np.array([float(r["sigma"]) for r in rows])
        corr = np.array([float(r["pearson_r"]) for r in rows])
        ax.plot(sig, corr, color=color, marker="o")
    ax.set_title("Seam recovery correlation")
    ax.set_xlabel("sigma")
    ax.set_ylabel("Pearson r")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    print("\nRendering sweep summary visualization...")
    plt.show()


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
    """Generates a random 2D mesh graph.

    For smaller n, we use a Delaunay triangulation. For larger n, we fall back to a
    k-NN graph (much more scalable in Colab environments).
    """
    points = rng.random((n_points, 2))

    # Delaunay becomes expensive/fragile at large n; use k-NN graph instead.
    if n_points <= DELAUNAY_MAX_N:
        tri = Delaunay(points)
        edges = set()
        for simplex in tri.simplices:
            a, b, c = (int(simplex[0]), int(simplex[1]), int(simplex[2]))
            edges.add((min(a, b), max(a, b)))
            edges.add((min(b, c), max(b, c)))
            edges.add((min(c, a), max(c, a)))
        edges = np.array(list(edges), dtype=np.int64)
        faces = tri.simplices
        boundary_vertices = np.unique(tri.convex_hull.reshape(-1)).astype(np.int64)
        return points, edges, faces, boundary_vertices

    # k-NN fallback: undirected edges to k nearest neighbors.
    tree = cKDTree(points)
    nn = tree.query(points, k=KNN_K + 1)[1][:, 1:]  # drop self
    u = np.repeat(np.arange(n_points, dtype=np.int64), KNN_K)
    v = nn.reshape(-1).astype(np.int64)
    a = np.minimum(u, v)
    b = np.maximum(u, v)
    pairs = np.stack([a, b], axis=1)
    edges = np.unique(pairs, axis=0)
    faces = None
    boundary_vertices = None
    return points, edges, faces, boundary_vertices


def solve_inverse_seam(points: np.ndarray, edges: np.ndarray, w_star: np.ndarray, stats: dict | None = None):
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

    # Small-N path: solve the constrained least-squares problem robustly via dense NNLS.
    # This avoids pathological behavior from solving normal equations on nearly singular
    # systems and avoids relying on sparse bounded solvers that can be brittle across
    # SciPy builds.
    if N <= 2000:
        E = len(edges)
        rows = np.concatenate([np.arange(E, dtype=np.int64), np.arange(E, dtype=np.int64)])
        cols = np.concatenate([u, v])
        data = np.concatenate([alpha, alpha])
        A_dense = np.zeros((len(edges), N), dtype=float)
        A_dense[rows, cols] = data
        x_nnls, _ = nnls(A_dense, w_star)
        if stats is not None:
            stats.update(
                {
                    "solver": "nnls",
                    "used_fallback": True,
                    "unconstrained_min": float("nan"),
                    "unconstrained_max": float("nan"),
                    "floor": 1e-12,
                    # NNLS enforces nonnegativity; zeros correspond to active constraints.
                    "active_frac": float(np.mean(x_nnls <= 1e-14)),
                }
            )
        X_opt = np.clip(x_nnls, 1e-12, None)
        if stats is not None:
            stats["floor_frac"] = float(np.mean(X_opt <= 1.0000000001e-12))
        return X_opt, l0, u, v

    # Primary path: solve normal equations (fast). Add a tiny ridge for numerical
    # stability without materially changing the objective.
    ridge = 1e-12 * max(1.0, float(np.mean(H_diag_vals)))
    if ridge > 0:
        H = H + sp.eye(N, format="csr") * ridge

    X_opt = spsolve(H, b)
    if stats is not None and np.all(np.isfinite(X_opt)):
        stats.update(
            {
                "solver": "normal_eq",
                "used_fallback": False,
                "unconstrained_min": float(np.min(X_opt)),
                "unconstrained_max": float(np.max(X_opt)),
                "floor": 1e-12,
            }
        )

    # Decide whether we trust the unconstrained solve.
    #
    # Important: we *always* clip X to stay positive before taking logs, but
    # clipping can severely degrade the least-squares optimum if the unconstrained
    # minimizer contains negatives/outliers. In that case we should instead solve
    # the constrained problem (X >= 0) directly.
    need_fallback = False
    if not np.all(np.isfinite(X_opt)):
        need_fallback = True
    else:
        min_x = float(np.min(X_opt))
        x_pos = X_opt[X_opt > 0]
        x_scale = float(np.median(x_pos)) if x_pos.size else 1.0
        x_scale = max(1e-12, x_scale)
        max_x = float(np.max(X_opt))

        # Negativity: allow only truly tiny negatives relative to a typical scale.
        if min_x <= -1e-10 * x_scale:
            need_fallback = True

        # Extreme outliers strongly suggest ill-conditioning; prefer constrained solve.
        if not need_fallback and max_x >= 1e6 * x_scale:
            need_fallback = True

        # If we would clip, verify that clipping would not meaningfully worsen the
        # objective. If it does, solve the constrained problem instead.
        if not need_fallback and min_x <= 0.0:
            X_clip = np.clip(X_opt, 1e-12, None)
            f_opt, _, _ = _objective_and_derivatives(N, u, v, alpha, w_star, X_opt)
            f_clip, _, _ = _objective_and_derivatives(N, u, v, alpha, w_star, X_clip)
            if (not np.isfinite(f_opt)) or (not np.isfinite(f_clip)):
                need_fallback = True
            elif f_clip > f_opt * (1.0 + 1e-6):
                need_fallback = True
            else:
                X_opt = X_clip

        # If the achieved fit is catastrophically bad, prefer the constrained solve.
        if not need_fallback:
            r = alpha * (X_opt[u] + X_opt[v]) - w_star
            denom = float(np.linalg.norm(w_star))
            rel_fit = float(np.linalg.norm(r) / (denom + 1e-12))
            if (not np.isfinite(rel_fit)) or rel_fit > 0.5:
                need_fallback = True

    if need_fallback:
        # Large-N fallback: constrained solve via sparse bounded least squares.
        E = len(edges)
        rows = np.concatenate([np.arange(E, dtype=np.int64), np.arange(E, dtype=np.int64)])
        cols = np.concatenate([u, v])
        data = np.concatenate([alpha, alpha])
        A = sp.coo_matrix((data, (rows, cols)), shape=(len(edges), N)).tocsr()
        res = lsq_linear(A, w_star, bounds=(0.0, np.inf), method="trf")
        if not res.success:
            raise RuntimeError(f"lsq_linear failed: {res.message}")
        X_opt = res.x
        if stats is not None:
            stats.update(
                {
                    "solver": "lsq_linear",
                    "used_fallback": True,
                    # In bounded least-squares, x==0 indicates an active lower bound.
                    "active_frac": float(np.mean(X_opt <= 1e-14)),
                }
            )

    X_opt = np.clip(X_opt, 1e-12, None)
    if stats is not None:
        stats["floor_frac"] = float(np.mean(X_opt <= 1.0000000001e-12))
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

    points, edges, faces, boundary_vertices = generate_mesh(n_points=n_points, rng=rng_points)
    N = len(points)
    E = len(edges)

    u = edges[:, 0].astype(np.int64)
    v = edges[:, 1].astype(np.int64)
    l0 = np.linalg.norm(points[u] - points[v], axis=1)

    s_gt = seam_gt(points)
    X_gt = np.exp(s_gt)
    w_gt = l0 * (X_gt[u] + X_gt[v]) / 2.0

    # Use a fixed noise direction xi and scale it by sigma for fair sweeps.
    xi = rng_noise.standard_normal(E)
    w_star = w_gt * (1.0 + sigma * xi)
    w_star = np.clip(w_star, 1e-6, None)

    solver_stats: dict[str, float | str | bool] = {}
    X_opt, _l0_check, _u_check, _v_check = solve_inverse_seam(points, edges, w_star, stats=solver_stats)
    s_opt = np.log(X_opt)
    w_opt = l0 * (X_opt[u] + X_opt[v]) / 2.0

    # Curvature error (only defined for triangle meshes).
    if faces is not None:
        w_gt_map = {(int(a), int(b)): float(w) for (a, b), w in zip(edges, w_gt)}
        w_opt_map = {(int(a), int(b)): float(w) for (a, b), w in zip(edges, w_opt)}
        K_gt, _ = compute_discrete_gaussian_curvature(N, faces, boundary_vertices, w_gt_map)
        K_opt, _ = compute_discrete_gaussian_curvature(N, faces, boundary_vertices, w_opt_map)
        curv_metrics = curvature_relative_error(K_gt, K_opt)
    else:
        curv_metrics = {
            "curv_rel_l2": float("nan"),
            "curv_abs_rel_mean": float("nan"),
            "curv_abs_rel_p95": float("nan"),
            "curv_abs_rel_max": float("nan"),
        }

    # Shortest-path distortion statistics between recovered/noisy metric and ground truth.
    # We sample a set of (source, target) pairs for speed.
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])

    A_gt = sp.coo_matrix((np.concatenate([w_gt, w_gt]), (rows, cols)), shape=(N, N)).tocsr()
    A_star = sp.coo_matrix((np.concatenate([w_star, w_star]), (rows, cols)), shape=(N, N)).tocsr()
    A_opt = sp.coo_matrix((np.concatenate([w_opt, w_opt]), (rows, cols)), shape=(N, N)).tocsr()

    n_sources = min(SAMPLE_SOURCES, N)
    sources = rng_pairs.choice(N, size=n_sources, replace=False)

    rel_errors_noisy = []
    rel_errors_opt = []
    distortions_noisy = []
    distortions_opt = []
    for src in sources:
        # Avoid trivial targets (src itself) and unreachable nodes (shouldn't happen for Delaunay graphs).
        candidates = np.arange(N)
        candidates = candidates[candidates != src]
        n_t = min(SAMPLE_TARGETS_PER_SOURCE, len(candidates))
        targets = rng_pairs.choice(candidates, size=n_t, replace=False)

        # Compute distances one source at a time to avoid allocating huge (n_sources x N) matrices.
        dgt_all = dijkstra(A_gt, directed=False, indices=int(src))
        dstar_all = dijkstra(A_star, directed=False, indices=int(src))
        dopt_all = dijkstra(A_opt, directed=False, indices=int(src))

        dgt = dgt_all[targets]
        dstar = dstar_all[targets]
        dopt = dopt_all[targets]
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
    # Track positivity-constraint activity for reporting.
    metrics.update(
        {
            "solver": str(solver_stats.get("solver", "unknown")),
            "used_fallback": bool(solver_stats.get("used_fallback", False)),
            "X_unconstrained_min": float(solver_stats.get("unconstrained_min", float("nan"))),
            "X_unconstrained_max": float(solver_stats.get("unconstrained_max", float("nan"))),
            "X_floor_frac": float(solver_stats.get("floor_frac", float("nan"))),
            "X_active_frac": float(solver_stats.get("active_frac", float("nan"))),
        }
    )
    metrics.update(curv_metrics)
    a_fit, b_fit, r2_fit = compute_affine_fit(s_gt, s_opt)
    metrics.update({"a_fit": float(a_fit), "b_fit": float(b_fit), "r2_fit": float(r2_fit)})
    return points, edges, s_gt, s_opt, w_gt, w_star, w_opt, metrics


def build_continuous_grid_geodesic_graph(grid_res: int):
    """High-resolution grid approximation to geodesics under g=e^{2 s_gt} g_Euclidean.

    Uses an 8-neighbor grid (adds diagonals) with edge lengths approximated by an
    endpoint-average rule:
        L(p,q) ≈ 0.5*(e^{s(p)} + e^{s(q)}) * ||p-q||.

    The diagonal connections substantially reduce anisotropy versus a 4-neighbor grid.
    """
    if grid_res < 10:
        raise ValueError("grid_res too small")

    xs = np.linspace(0.0, 1.0, grid_res)
    ys = np.linspace(0.0, 1.0, grid_res)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    s = seam_gt(pts)
    f = np.exp(s)

    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])

    # Horizontal edges (i,j) -> (i+1,j)
    n = grid_res
    idx = np.arange(n * n, dtype=np.int64).reshape(n, n)
    left = idx[:, :-1].reshape(-1)
    right = idx[:, 1:].reshape(-1)
    w_h = 0.5 * (f[left] + f[right]) * dx

    # Vertical edges (i,j) -> (i,j+1)
    down = idx[:-1, :].reshape(-1)
    up = idx[1:, :].reshape(-1)
    w_v = 0.5 * (f[down] + f[up]) * dy

    # Diagonal edges
    diag1_a = idx[:-1, :-1].reshape(-1)
    diag1_b = idx[1:, 1:].reshape(-1)
    diag2_a = idx[:-1, 1:].reshape(-1)
    diag2_b = idx[1:, :-1].reshape(-1)
    ddiag = float(np.hypot(dx, dy))
    w_d1 = 0.5 * (f[diag1_a] + f[diag1_b]) * ddiag
    w_d2 = 0.5 * (f[diag2_a] + f[diag2_b]) * ddiag

    rows = np.concatenate([left, right, down, up, diag1_a, diag1_b, diag2_a, diag2_b])
    cols = np.concatenate([right, left, up, down, diag1_b, diag1_a, diag2_b, diag2_a])
    data = np.concatenate([w_h, w_h, w_v, w_v, w_d1, w_d1, w_d2, w_d2]).astype(float)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n * n, n * n)).tocsr()
    return pts, A


def _generate_quasi_uniform_points(n_points: int, rng: np.random.Generator, jitter: float = 0.25) -> np.ndarray:
    """Quasi-uniform points in [0,1]^2 via a jittered grid.

    This is used for the convergence study to better match quasi-uniform triangulation
    assumptions (and to reduce run-to-run stochasticity).
    """
    if n_points < 4:
        raise ValueError("n_points too small")

    m = int(np.ceil(np.sqrt(n_points)))
    xs = np.linspace(0.0, 1.0, m)
    ys = np.linspace(0.0, 1.0, m)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)

    # Jitter amount is relative to grid spacing.
    if m >= 2:
        h = float(xs[1] - xs[0])
    else:
        h = 1.0
    noise = rng.uniform(low=-1.0, high=1.0, size=pts.shape) * (jitter * h)
    pts = np.clip(pts + noise, 0.0, 1.0)

    return pts[: int(n_points)]


def _nearest_grid_index(p: np.ndarray, grid_res: int) -> int:
    """Nearest node index in a [0,1]^2 regular grid (grid_res x grid_res)."""
    x = float(np.clip(p[0], 0.0, 1.0))
    y = float(np.clip(p[1], 0.0, 1.0))
    ix = int(np.rint(x * (grid_res - 1)))
    iy = int(np.rint(y * (grid_res - 1)))
    ix = int(np.clip(ix, 0, grid_res - 1))
    iy = int(np.clip(iy, 0, grid_res - 1))
    return iy * grid_res + ix


def run_convergence_rate_study(
    n_grid: list[int],
    test_points: np.ndarray,
    sigma: float = 0.0,
    grid_res: int = 250,
    seed: int = 123,
    n_repeats: int = 1,
    print_table: bool = True,
    plateau_fit_factor: float = 1.5,
):
    """Demonstrate O(h) convergence by comparing mesh graph distances to a
    high-resolution grid approximation of the continuous geodesic distance.
    """
    test_points = np.asarray(test_points, dtype=float)
    if test_points.ndim != 2 or test_points.shape[1] != 2:
        raise ValueError("test_points must have shape (m,2)")

    cont_pts, cont_A = build_continuous_grid_geodesic_graph(grid_res=grid_res)
    cont_src_idx = np.array([_nearest_grid_index(p, grid_res) for p in test_points], dtype=np.int64)
    # Compute all-pairs distances between test points on the continuous grid (few sources).
    cont_d_all = dijkstra(cont_A, directed=False, indices=cont_src_idx)

    pairs = [(i, j) for i in range(len(test_points)) for j in range(i + 1, len(test_points))]
    cont_d_pairs = np.array([float(cont_d_all[i, cont_src_idx[j]]) for i, j in pairs], dtype=float)

    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")

    rows = []
    for n_points in n_grid:
        mean_errs = []
        max_errs = []
        h_vals_rep = []
        for rep in range(n_repeats):
            # Different mesh realizations per repeat, deterministic.
            rng_points = np.random.default_rng(seed + 100_000 * rep + int(n_points))

            # For the convergence plot we intentionally use a quasi-uniform triangulation.
            points = _generate_quasi_uniform_points(n_points=n_points, rng=rng_points, jitter=0.25)
            tri = Delaunay(points)
            edges_set = set()
            for simplex in tri.simplices:
                a, b, c = (int(simplex[0]), int(simplex[1]), int(simplex[2]))
                edges_set.add((min(a, b), max(a, b)))
                edges_set.add((min(b, c), max(b, c)))
                edges_set.add((min(c, a), max(c, a)))
            edges = np.array(list(edges_set), dtype=np.int64)
            u = edges[:, 0].astype(np.int64)
            v = edges[:, 1].astype(np.int64)
            l0 = np.linalg.norm(points[u] - points[v], axis=1)
            s = seam_gt(points)
            X = np.exp(s)
            w = l0 * (X[u] + X[v]) / 2.0

            # Optional noise injection (kept for experimentation; default sigma=0).
            if sigma != 0.0:
                rng_noise = np.random.default_rng(seed + 999 + 10_000 * rep + int(n_points))
                xi = rng_noise.standard_normal(len(w))
                w = np.clip(w * (1.0 + sigma * xi), 1e-6, None)

            # Mesh graph distances between nearest vertices to the same test points.
            tree = cKDTree(points)
            mesh_src = np.array([int(tree.query(p)[1]) for p in test_points], dtype=np.int64)

            mesh_rows = np.concatenate([u, v])
            mesh_cols = np.concatenate([v, u])
            A = sp.coo_matrix(
                (np.concatenate([w, w]), (mesh_rows, mesh_cols)),
                shape=(len(points), len(points)),
            ).tocsr()

            mesh_d_all = dijkstra(A, directed=False, indices=mesh_src)
            mesh_d_pairs = np.array([float(mesh_d_all[i, mesh_src[j]]) for i, j in pairs], dtype=float)

            abs_err = np.abs(mesh_d_pairs - cont_d_pairs)
            mean_errs.append(float(np.mean(abs_err)))
            max_errs.append(float(np.max(abs_err)))
            h_vals_rep.append(float(np.mean(l0)))

        if not mean_errs:
            continue

        mean_abs_err = float(np.mean(mean_errs))
        mean_abs_err_std = float(np.std(mean_errs, ddof=1)) if len(mean_errs) >= 2 else 0.0
        max_abs_err = float(np.mean(max_errs))
        max_abs_err_std = float(np.std(max_errs, ddof=1)) if len(max_errs) >= 2 else 0.0
        h = float(np.mean(h_vals_rep))
        rows.append({
            "n_points": int(n_points),
            "h": h,
            "mean_abs_err": mean_abs_err,
            "mean_abs_err_std": mean_abs_err_std,
            "max_abs_err": max_abs_err,
            "max_abs_err_std": max_abs_err_std,
            "n_repeats": int(n_repeats),
        })

    if not rows:
        return

    rows = sorted(rows, key=lambda r: r["h"], reverse=True)
    h_vals = np.array([r["h"] for r in rows], dtype=float)
    err_vals = np.array([r["mean_abs_err"] for r in rows], dtype=float)
    err_std = np.array([r.get("mean_abs_err_std", 0.0) for r in rows], dtype=float)

    mask = (h_vals > 0) & (err_vals > 0) & np.isfinite(h_vals) & np.isfinite(err_vals)
    slope_full = float("nan")
    intercept_full = float("nan")
    if np.count_nonzero(mask) >= 2:
        slope_full, intercept_full = np.polyfit(np.log(h_vals[mask]), np.log(err_vals[mask]), 1)
        slope_full = float(slope_full)
        intercept_full = float(intercept_full)

    # Robust fit: exclude the near-plateau region where err is close to its minimum.
    slope = float("nan")
    intercept = float("nan")
    if np.count_nonzero(mask) >= 2:
        min_err = float(np.min(err_vals[mask]))
        thresh = float(max(0.0, plateau_fit_factor) * min_err)
        mask2 = mask & (err_vals >= thresh)
        if np.count_nonzero(mask2) >= 2:
            slope, intercept = np.polyfit(np.log(h_vals[mask2]), np.log(err_vals[mask2]), 1)
            slope = float(slope)
            intercept = float(intercept)

    if print_table:
        print("\n--- Convergence study data (discrete vs grid-geodesic) ---")
        print(f"grid_res={grid_res}, sigma={sigma:.3f}, repeats={n_repeats}")
        if np.isfinite(slope_full):
            print(f"log-log fit slope (all points): {slope_full:.3f}")
        if np.isfinite(slope):
            print(f"log-log fit slope (excluding plateau; factor={plateau_fit_factor:.2f}): {slope:.3f}")
        print("n_points      h(mean)     mean_abs_err    std")
        for r in rows:
            print(
                f"{r['n_points']:7d}  {r['h']:.6f}   {r['mean_abs_err']:.6e}  {r['mean_abs_err_std']:.2e}"
            )

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.0))
    if n_repeats >= 2 and np.any(err_std > 0):
        ax.errorbar(h_vals, err_vals, yerr=err_std, fmt="o", capsize=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        ax.loglog(h_vals, err_vals, marker="o")
    ax.set_xlabel("h (mean Euclidean edge length)")
    ax.set_ylabel("mean |d_n - d_g| over test pairs")
    title = "Convergence of discrete shortest-path distances"
    if np.isfinite(slope):
        title += f" (fit slope ≈ {slope:.2f})"
    elif np.isfinite(slope_full):
        title += f" (fit slope ≈ {slope_full:.2f})"
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    print("\nRendering convergence rate plot...")
    plt.show()

    return rows, slope

def main():
    n_grid = [100, 1000, 10000]
    sigma_grid = [0.00, 0.05, 0.10, 0.15, 0.20]

    print("--- Seam-Driven Geometry Validation (sweep) ---")
    print(
        "Columns: n  E  sigma  mean_l0  max_l0  fit(w_opt,w*)  noisy->truth  denoise->truth  "
        "path_rel_mean_noisy  path_rel_p95_noisy  path_rel_mean  path_rel_p95  "
        "path_rel_mean_improve  path_rel_p95_improve  curv_rel_l2  curv_abs_rel_mean  corr  a_fit  b_fit  R2"
    )
    results = []
    deriv_checks = []

    example_payload = None
    example_key = (800, 0.15)

    last_payload = None

    for n in n_grid:
        for sigma in sigma_grid:
            points, edges, s_gt, s_opt, w_gt, w_star, w_opt, m = run_experiment(n_points=n, sigma=sigma, seed=42)
            results.append(m)
            last_payload = (points, edges, s_gt, s_opt, m)

            chk = derivative_check(points, edges, w_star, seed=123)
            deriv_checks.append({"n": n, "sigma": float(sigma), **chk})

            print(
                f"{m['n_points']:4d} {m['n_edges']:5d} {m['sigma']:5.2f} "
                f"{m['mean_l0']:.4f} {m['max_l0']:.4f} "
                f"{m['fit_to_target']:.4f} {m['noisy_to_truth']:.4f} {m['denoised_to_truth']:.4f} "
                f"{m['path_rel_mean_noisy']:.4f} {m['path_rel_p95_noisy']:.4f} "
                f"{m['path_rel_mean']:.4f} {m['path_rel_p95']:.4f} "
                f"{m['path_rel_mean_improve']:.4f} {m['path_rel_p95_improve']:.4f} "
                f"{m['curv_rel_l2']:.4f} {m['curv_abs_rel_mean']:.4f} "
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

    plot_sweep_summary(results)

    # Always render a representative example figure.
    if example_payload is None:
        example_payload = last_payload

    if example_payload is not None:
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
        axes[2].plot(
            [lo, hi],
            [m['a_fit'] * lo + m['b_fit'], m['a_fit'] * hi + m['b_fit']],
            color='blue',
            lw=2,
        )
        axes[2].set_title(f"Seam Recovery Correlation (n={m['n_points']}, sigma={m['sigma']:.2f})")
        axes[2].set_xlabel("Ground Truth Seam Value")
        axes[2].set_ylabel("Recovered Seam Value")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        print("\nRendering example visualization...")
        plt.show()

    # Optional: explicit O(h) convergence plot for shortest-path distances.
    # Restricted to Delaunay meshes (n <= 5000) where faces are well-defined.
    if RUN_CONVERGENCE_STUDY:
        # Use near-square counts for quasi-uniform jittered-grid triangulations.
        conv_n_grid = [225, 400, 625, 900, 1600, 2500, 3600, 4900]
        conv_test_points = np.array(
            [
                [0.2, 0.2],
                [0.8, 0.2],
                [0.2, 0.8],
                [0.8, 0.8],
                [0.5, 0.5],
            ],
            dtype=float,
        )
        run_convergence_rate_study(
            n_grid=conv_n_grid,
            test_points=conv_test_points,
            sigma=0.0,
            grid_res=250,
            seed=123,
            n_repeats=1,
            print_table=True,
            plateau_fit_factor=1.5,
        )

if __name__ == "__main__":
    main()