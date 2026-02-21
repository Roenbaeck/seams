"""Generate paper figures for ExRelSeam.tex.

Creates three PDF+PNG figures in ./figures:
  - fig1_seam_curvature: seam field and induced discrete curvature on a small Delaunay mesh
  - fig2_convergence: log-log convergence of mean |d_n - d_g| vs h
  - fig3_inverse_design: histogram of edge-weight relative error before/after inverse design

This script is deterministic (fixed RNG seeds) to support reproducible builds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import Delaunay, cKDTree

import validation


FIG_DIR = Path(__file__).resolve().parent / "figures"


def _save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG_DIR / f"{stem}.pdf"
    png_path = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")


def figure1_seam_and_curvature(seed: int = 7, n_points: int = 500) -> None:
    rng = np.random.default_rng(seed)
    points = rng.random((n_points, 2))

    tri = Delaunay(points)
    faces = tri.simplices
    boundary_vertices = np.unique(tri.convex_hull.reshape(-1)).astype(np.int64)

    # Build undirected edge set from faces.
    edges = set()
    for (i, j, k) in faces:
        i = int(i)
        j = int(j)
        k = int(k)
        edges.add((min(i, j), max(i, j)))
        edges.add((min(j, k), max(j, k)))
        edges.add((min(k, i), max(k, i)))
    edges = np.array(sorted(edges), dtype=np.int64)

    u = edges[:, 0]
    v = edges[:, 1]
    l0 = np.linalg.norm(points[u] - points[v], axis=1)

    s = validation.seam_gt(points)
    X = np.exp(s)
    l_s = l0 * (X[u] + X[v]) / 2.0

    # Discrete Gaussian curvature (angle defect) computed from intrinsic edge lengths.
    edge_length_map = {(int(a), int(b)): float(L) for (a, b), L in zip(edges, l_s)}
    K, _angle_sum = validation.compute_discrete_gaussian_curvature(
        n_vertices=len(points),
        faces=faces,
        boundary_vertices=boundary_vertices,
        edge_length_map=edge_length_map,
    )

    boundary_mask = np.zeros(len(points), dtype=bool)
    boundary_mask[boundary_vertices] = True

    triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles=faces)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    tpc0 = axes[0].tripcolor(triang, s, shading="gouraud", cmap="viridis")
    axes[0].triplot(triang, color="black", linewidth=0.25, alpha=0.35)
    axes[0].set_title("Seam field $s$ on a Delaunay mesh")
    axes[0].set_aspect("equal")
    axes[0].axis("off")
    fig.colorbar(tpc0, ax=axes[0], fraction=0.046, pad=0.03)

    # Boundary curvature (pi - angle sum) can dominate the dynamic range on a planar patch.
    # For a more representative visualization, plot only interior angle-defect values.
    K_plot = K.astype(float).copy()
    K_plot[boundary_mask] = np.nan

    # Clip extreme interior curvature values for a readable colormap.
    finite_K = K_plot[np.isfinite(K_plot)]
    if finite_K.size:
        clip = float(np.percentile(np.abs(finite_K), 98))
    else:
        clip = 1.0
    K_clip = np.clip(K_plot, -clip, clip)

    tpc1 = axes[1].tripcolor(triang, K_clip, shading="gouraud", cmap="coolwarm")
    axes[1].triplot(triang, color="black", linewidth=0.25, alpha=0.35)
    axes[1].set_title("Angle-defect curvature $K_s$ induced by $\ell_s$ (boundary masked)")
    axes[1].set_aspect("equal")
    axes[1].axis("off")
    fig.colorbar(tpc1, ax=axes[1], fraction=0.046, pad=0.03)

    plt.tight_layout()
    _save(fig, "fig1_seam_curvature")
    plt.close(fig)


@dataclass(frozen=True)
class ConvergenceRow:
    n_points: int
    h: float
    mean_abs_err: float


def _convergence_data(
    n_grid: list[int],
    test_points: np.ndarray,
    grid_res: int = 250,
    seed: int = 123,
) -> list[ConvergenceRow]:
    test_points = np.asarray(test_points, dtype=float)

    # Continuous proxy (grid geodesic).
    _cont_pts, cont_A = validation.build_continuous_grid_geodesic_graph(grid_res=grid_res)
    cont_src_idx = np.array([validation._nearest_grid_index(p, grid_res) for p in test_points], dtype=np.int64)
    cont_d_all = dijkstra(cont_A, directed=False, indices=cont_src_idx)

    pairs = [(i, j) for i in range(len(test_points)) for j in range(i + 1, len(test_points))]
    cont_d_pairs = np.array([float(cont_d_all[i, cont_src_idx[j]]) for i, j in pairs], dtype=float)

    rows: list[ConvergenceRow] = []
    for n_points in n_grid:
        rng_points = np.random.default_rng(seed + int(n_points))
        points = validation._generate_quasi_uniform_points(n_points=n_points, rng=rng_points, jitter=0.25)
        tri = Delaunay(points)

        edges_set: set[tuple[int, int]] = set()
        for simplex in tri.simplices:
            a, b, c = (int(simplex[0]), int(simplex[1]), int(simplex[2]))
            edges_set.add((min(a, b), max(a, b)))
            edges_set.add((min(b, c), max(b, c)))
            edges_set.add((min(c, a), max(c, a)))
        edges = np.array(list(edges_set), dtype=np.int64)

        u = edges[:, 0].astype(np.int64)
        v = edges[:, 1].astype(np.int64)
        l0 = np.linalg.norm(points[u] - points[v], axis=1)

        s = validation.seam_gt(points)
        X = np.exp(s)
        w = l0 * (X[u] + X[v]) / 2.0

        # Mesh distances between nearest vertices to the test points.
        tree = cKDTree(points)
        mesh_src = np.array([int(tree.query(p)[1]) for p in test_points], dtype=np.int64)

        mesh_rows = np.concatenate([u, v])
        mesh_cols = np.concatenate([v, u])
        A = sp.coo_matrix((np.concatenate([w, w]), (mesh_rows, mesh_cols)), shape=(len(points), len(points))).tocsr()
        mesh_d_all = dijkstra(A, directed=False, indices=mesh_src)
        mesh_d_pairs = np.array([float(mesh_d_all[i, mesh_src[j]]) for i, j in pairs], dtype=float)

        abs_err = np.abs(mesh_d_pairs - cont_d_pairs)
        rows.append(ConvergenceRow(n_points=int(n_points), h=float(np.mean(l0)), mean_abs_err=float(np.mean(abs_err))))

    rows.sort(key=lambda r: r.h, reverse=True)
    return rows


def _fit_loglog_slope(h: np.ndarray, err: np.ndarray, plateau_fit_factor: float = 1.5) -> float:
    mask = (h > 0) & (err > 0) & np.isfinite(h) & np.isfinite(err)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    min_err = float(np.min(err[mask]))
    thresh = float(max(0.0, plateau_fit_factor) * min_err)
    mask2 = mask & (err >= thresh)
    if np.count_nonzero(mask2) < 2:
        mask2 = mask
    slope, _intercept = np.polyfit(np.log(h[mask2]), np.log(err[mask2]), 1)
    return float(slope)


def _fit_loglog_line(
    h: np.ndarray, err: np.ndarray, plateau_fit_factor: float = 1.5
) -> tuple[float, float, np.ndarray]:
    mask = (h > 0) & (err > 0) & np.isfinite(h) & np.isfinite(err)
    if np.count_nonzero(mask) < 2:
        return float("nan"), float("nan"), mask
    min_err = float(np.min(err[mask]))
    thresh = float(max(0.0, plateau_fit_factor) * min_err)
    mask2 = mask & (err >= thresh)
    if np.count_nonzero(mask2) < 2:
        mask2 = mask
    slope, intercept = np.polyfit(np.log(h[mask2]), np.log(err[mask2]), 1)
    return float(slope), float(intercept), mask2


def figure2_convergence(seed: int = 123) -> None:
    n_grid = [225, 400, 625, 900, 1600, 2500, 3600, 4900]
    test_points = np.array(
        [[0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8], [0.5, 0.5]], dtype=float
    )

    rows = _convergence_data(n_grid=n_grid, test_points=test_points, grid_res=250, seed=seed)
    h = np.array([r.h for r in rows], dtype=float)
    err = np.array([r.mean_abs_err for r in rows], dtype=float)
    slope, intercept, fit_mask = _fit_loglog_line(h, err, plateau_fit_factor=1.5)

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 5.0))
    ax.loglog(h, err, marker="o", label="data")

    if np.isfinite(slope) and np.isfinite(intercept):
        h_line = np.array([np.max(h), np.min(h)], dtype=float)
        err_line = np.exp(intercept) * (h_line**slope)
        ax.loglog(h_line, err_line, linestyle="--", linewidth=1.5, label=rf"fit slope {slope:.2f}")

        h_fit = h[fit_mask]
        err_fit = err[fit_mask]
        if len(h_fit):
            pivot = int(len(h_fit) // 2)
            h0 = float(h_fit[pivot])
            err0 = float(err_fit[pivot])
            err_ref = err0 * (h_line / h0)
            ax.loglog(h_line, err_ref, linestyle=":", linewidth=1.5, label="reference slope 1")
    ax.set_xlabel("$h$ (mean Euclidean edge length)")
    ax.set_ylabel(r"mean $|d_n - d_g|$ over test pairs")
    title = "Convergence of discrete shortest-path distances"
    if np.isfinite(slope):
        title += rf" (fit slope $\approx$ {slope:.2f})"
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    _save(fig, "fig2_convergence")
    plt.close(fig)


def figure3_inverse_design_hist(seed: int = 42, n_points: int = 10_000, sigma: float = 0.10) -> None:
    # Run one experiment and extract edge weights.
    points, edges, s_gt, s_opt, w_gt, w_star, w_opt, metrics = validation.run_experiment(
        n_points=int(n_points), sigma=float(sigma), seed=int(seed)
    )

    rel_noisy = np.abs(w_star - w_gt) / (np.abs(w_gt) + 1e-12)
    rel_opt = np.abs(w_opt - w_gt) / (np.abs(w_gt) + 1e-12)

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.8))
    hi = float(np.percentile(rel_noisy, 99.5))
    lo = float(min(np.min(rel_noisy), np.min(rel_opt)))
    bins = np.logspace(np.log10(lo), np.log10(hi), 50)
    ax.hist(rel_noisy, bins=bins, alpha=0.55, label=r"noisy $w^*$")
    ax.hist(rel_opt, bins=bins, alpha=0.55, label=r"recovered $w_{\mathrm{opt}}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"relative edge-weight error $|w - w_{\mathrm{gt}}|/|w_{\mathrm{gt}}|$")
    ax.set_ylabel("count")
    ax.set_title(rf"Inverse design denoising (n={int(metrics['n_points'])}, $\sigma$={metrics['sigma']:.2f})")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    _save(fig, "fig3_inverse_design")
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    figure1_seam_and_curvature()
    figure2_convergence()
    figure3_inverse_design_hist()
    dt = time.perf_counter() - t0
    print(f"Wrote figures to {FIG_DIR} in {dt:.2f}s")


if __name__ == "__main__":
    main()
