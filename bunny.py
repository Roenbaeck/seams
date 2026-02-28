"""
Stanford Bunny 3D Demonstration of Inverse Seam Design

Validates the metric nearness projection (Theorem 7) and its
Laplacian-regularised extension on a real 3D triangle mesh — the
Stanford Bunny — to complement the synthetic 2D tests in validation.py.

Experiment
----------
1. Load the Stanford Bunny mesh (~35 k vertices, ~70 k faces).
2. Define a smooth ground-truth seam field s_gt over the surface.
3. Compute exact conformal edge weights  w_gt = ℓ₀ · (X_u + X_v) / 2.
4. Corrupt with multiplicative Gaussian noise: w* = w_gt · (1 + σξ).
5. Solve the baseline QP (Theorem 7) and the Laplacian-regularised QP
   to recover X_opt, hence s_opt and w_opt for each variant.
6. Produce two publication figures (mesh panels + histogram) and print summary metrics.

Dependencies: numpy, scipy, matplotlib (same as the main validation).
The Stanford Bunny OBJ is downloaded automatically on first run.
"""

from __future__ import annotations

import time
from pathlib import Path
import urllib.request

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from scipy.optimize import lsq_linear

import validation

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIG_DIR = Path(__file__).resolve().parent / "figures"
BUNNY_PATH = Path(__file__).resolve().parent / "stanford-bunny.obj"
BUNNY_URL = (
    "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/"
    "master/data/stanford-bunny.obj"
)


# ---------------------------------------------------------------------------
# Mesh I/O
# ---------------------------------------------------------------------------
def download_bunny() -> Path:
    """Download the Stanford Bunny OBJ (cached after first call)."""
    if BUNNY_PATH.exists():
        print(f"Using cached {BUNNY_PATH.name}")
        return BUNNY_PATH
    print(f"Downloading Stanford Bunny from {BUNNY_URL} ...")
    urllib.request.urlretrieve(BUNNY_URL, BUNNY_PATH)
    print(f"Saved to {BUNNY_PATH}")
    return BUNNY_PATH


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal OBJ parser.  Returns (vertices [N,3], faces [F,3])."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) >= 4:
                    # Fan-triangulate polygonal faces.
                    for k in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[k], idx[k + 1]])
    return np.array(vertices, dtype=float), np.array(faces, dtype=np.int64)


def extract_edges(faces: np.ndarray) -> np.ndarray:
    """Unique undirected edges from a triangle array.  Returns [E, 2]."""
    edge_set: set[tuple[int, int]] = set()
    for i, j, k in faces:
        i, j, k = int(i), int(j), int(k)
        edge_set.add((min(i, j), max(i, j)))
        edge_set.add((min(j, k), max(j, k)))
        edge_set.add((min(k, i), max(k, i)))
    return np.array(sorted(edge_set), dtype=np.int64)


def largest_component(
    vertices: np.ndarray, faces: np.ndarray, edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only the largest connected component of the mesh.

    The Stanford Bunny OBJ contains ~1 100 orphan vertices (unused by any
    face).  These have no edge constraints and would be unconstrained in the
    QP, producing garbage X values that corrupt the seam correlation.
    """
    N = len(vertices)
    u, v = edges[:, 0], edges[:, 1]
    A = sp.coo_matrix(
        (np.ones(2 * len(edges)), (np.r_[u, v], np.r_[v, u])),
        shape=(N, N),
    )
    n_comp, labels = connected_components(A, directed=False)
    if n_comp == 1:
        return vertices, faces, edges

    # Identify and keep the largest component.
    comp_sizes = np.bincount(labels)
    main_label = int(np.argmax(comp_sizes))
    keep = labels == main_label

    # Reindex vertices.
    old_to_new = -np.ones(N, dtype=np.int64)
    old_to_new[keep] = np.arange(int(keep.sum()), dtype=np.int64)
    new_verts = vertices[keep]

    # Reindex faces (drop faces touching removed vertices).
    face_keep = keep[faces].all(axis=1)
    new_faces = old_to_new[faces[face_keep]]

    # Reindex edges.
    edge_keep = keep[edges].all(axis=1)
    new_edges = old_to_new[edges[edge_keep]]

    n_removed = N - int(keep.sum())
    if n_removed > 0:
        print(f"  Removed {n_removed:,} orphan vertices ({n_comp} components → 1).")
    return new_verts, new_faces, new_edges


# ---------------------------------------------------------------------------
# Seam field
# ---------------------------------------------------------------------------
def seam_gt_3d(vertices: np.ndarray) -> np.ndarray:
    """Smooth ground-truth seam on a 3D surface.

    Uses a low-frequency height-based pattern that is well-resolved by the
    mesh and produces a visually distinctive gradient on the bunny.
    """
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    scale = v_max - v_min
    scale[scale < 1e-12] = 1.0
    v_n = (vertices - v_min) / scale
    # Gentle height gradient + lateral modulation (single period each axis).
    return (
        0.5 * np.sin(np.pi * v_n[:, 1])
        + 0.25 * np.cos(np.pi * v_n[:, 0])
    )


# ---------------------------------------------------------------------------
# Triangle-inequality check  (Review point §2)
# ---------------------------------------------------------------------------
def triangle_inequality_check(
    faces: np.ndarray, edges: np.ndarray, weights: np.ndarray
) -> tuple[int, int]:
    """Count faces where the edge weights violate the strict triangle inequality.

    Returns (n_violations, n_faces).
    """
    edge_to_idx: dict[tuple[int, int], int] = {}
    for idx, (eu, ev) in enumerate(edges):
        edge_to_idx[(int(eu), int(ev))] = idx

    F = len(faces)
    a = np.empty(F)
    b = np.empty(F)
    c = np.empty(F)
    for fi in range(F):
        i, j, k = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        a[fi] = weights[edge_to_idx[(min(i, j), max(i, j))]]
        b[fi] = weights[edge_to_idx[(min(j, k), max(j, k))]]
        c[fi] = weights[edge_to_idx[(min(k, i), max(k, i))]]

    violations = ~((a + b > c) & (b + c > a) & (c + a > b))
    return int(violations.sum()), F


def face_edge_indices(faces: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return edge indices (a,b,c) for each face's three edges."""
    edge_to_idx: dict[tuple[int, int], int] = {}
    for idx, (eu, ev) in enumerate(edges):
        edge_to_idx[(int(eu), int(ev))] = idx

    F = len(faces)
    a = np.empty(F, dtype=np.int64)
    b = np.empty(F, dtype=np.int64)
    c = np.empty(F, dtype=np.int64)
    for fi in range(F):
        i, j, k = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        a[fi] = edge_to_idx[(min(i, j), max(i, j))]
        b[fi] = edge_to_idx[(min(j, k), max(j, k))]
        c[fi] = edge_to_idx[(min(k, i), max(k, i))]
    return a, b, c


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals from triangle faces."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)  # length = 2 × area
    vnormals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(vnormals, faces[:, k], face_normals)
    nlen = np.linalg.norm(vnormals, axis=1, keepdims=True)
    nlen[nlen < 1e-15] = 1.0
    return vnormals / nlen


def displace_along_normals(
    vertices: np.ndarray, normals: np.ndarray, scalar: np.ndarray, scale: float,
) -> np.ndarray:
    """Displace vertices along their normals by `scale * scalar`."""
    return vertices + normals * (scale * scalar)[:, np.newaxis]


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _rotation_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Combined Y-rotation (azimuth) then X-rotation (elevation)."""
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    Ry = np.array(
        [[np.cos(az), 0, np.sin(az)], [0, 1, 0], [-np.sin(az), 0, np.cos(az)]]
    )
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]]
    )
    return Rx @ Ry


def render_mesh(
    ax: plt.Axes,
    vertices: np.ndarray,
    faces: np.ndarray,
    scalar: np.ndarray,
    title: str,
    cmap_name: str = "viridis",
    azimuth: float = 150.0,
    elevation: float = 15.0,
    vmin: float | None = None,
    vmax: float | None = None,
    per_face: bool = False,
) -> tuple[Normalize, object]:
    """Depth-sorted, Lambertian-shaded orthographic rendering of a 3D mesh.

    Parameters
    ----------
    scalar : array, shape (N_vertices,) or (N_faces,)
        Per-vertex values are averaged to faces.  Set *per_face=True*
        when providing one value per face directly.
    """

    R = _rotation_matrix(azimuth, elevation)
    rotated = vertices @ R.T

    # Orthographic projection
    xy = rotated[:, :2]
    depth = rotated[:, 2]

    # Face normals in camera space (for shading)
    rv0, rv1, rv2 = rotated[faces[:, 0]], rotated[faces[:, 1]], rotated[faces[:, 2]]
    normals = np.cross(rv1 - rv0, rv2 - rv0)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    nlen[nlen < 1e-15] = 1.0
    normals /= nlen

    # Simple directional light + ambient
    light = np.array([0.2, 0.3, 1.0])
    light /= np.linalg.norm(light)
    shade = np.abs(normals @ light)
    shade = 0.35 + 0.65 * shade

    # Painter's algorithm: back-to-front
    face_depth = depth[faces].mean(axis=1)
    order = np.argsort(face_depth)
    sorted_faces = faces[order]
    sorted_shade = shade[order]

    # Colormap
    if vmin is None:
        vmin = float(scalar.min())
    if vmax is None:
        vmax = float(scalar.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps[cmap_name]

    if per_face:
        face_scalar = scalar[order]
    else:
        face_scalar = scalar[sorted_faces].mean(axis=1)
    colors = cmap(norm(face_scalar)).copy()
    colors[:, :3] *= sorted_shade[:, np.newaxis]
    colors = np.clip(colors, 0.0, 1.0)

    face_xy = xy[sorted_faces]
    poly = PolyCollection(
        face_xy, facecolors=colors, edgecolors="none", linewidths=0, rasterized=True
    )
    ax.add_collection(poly)

    margin = (xy[:, 0].max() - xy[:, 0].min()) * 0.03
    ax.set_xlim(xy[:, 0].min() - margin, xy[:, 0].max() + margin)
    ax.set_ylim(xy[:, 1].min() - margin, xy[:, 1].max() + margin)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.axis("off")

    return norm, cmap


def _save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG_DIR / f"{stem}.pdf"
    png_path = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Laplacian-regularised inverse-design solver
# ---------------------------------------------------------------------------
def solve_inverse_seam_regularised(
    vertices: np.ndarray,
    edges: np.ndarray,
    w_star: np.ndarray,
    reg_mu: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """Solve the Laplacian-regularised inverse-design QP.

    Minimises  ½ X^T (H + μ L) X − b^T X  subject to  X ≥ 0,
    where H is the signless-Laplacian Hessian of the unweighted
    edge-fitting objective and L is the combinatorial graph Laplacian.

    Returns (X_opt, solve_time).
    """
    N = len(vertices)
    E = len(edges)
    u = edges[:, 0].astype(np.int64)
    v = edges[:, 1].astype(np.int64)
    l0 = np.linalg.norm(vertices[u] - vertices[v], axis=1)
    alpha = l0 / 2.0
    alpha_sq = alpha ** 2

    # Signless-Laplacian Hessian H
    I = np.concatenate([u, v])
    J = np.concatenate([v, u])
    V_off = np.concatenate([alpha_sq, alpha_sq])
    H = sp.coo_matrix((V_off, (I, J)), shape=(N, N))
    H_diag = (np.bincount(u, weights=alpha_sq, minlength=N)
              + np.bincount(v, weights=alpha_sq, minlength=N))
    H = (H + sp.diags(H_diag)).tocsr()

    # RHS vector b
    b_edge = alpha * w_star
    b = (np.bincount(u, weights=b_edge, minlength=N)
         + np.bincount(v, weights=b_edge, minlength=N))

    # Combinatorial graph Laplacian L = D − A
    ones_E = np.ones(E)
    A_adj = sp.coo_matrix(
        (np.concatenate([ones_E, ones_E]),
         (np.concatenate([u, v]), np.concatenate([v, u]))),
        shape=(N, N),
    )
    deg = np.bincount(u, minlength=N) + np.bincount(v, minlength=N)
    L = sp.diags(deg.astype(float)) - A_adj.tocsr()

    H_reg = H + reg_mu * L.tocsr()

    # Tiny ridge for numerical stability
    ridge = 1e-12 * max(1.0, float(H_reg.diagonal().mean()))
    H_reg = H_reg + sp.eye(N, format="csr") * ridge

    t1 = time.perf_counter()
    X = spsolve(H_reg, b)
    dt = time.perf_counter() - t1

    # Positivity fallback via sparse bounded least-squares
    if np.any(~np.isfinite(X)) or np.min(X) < -1e-8 * max(1e-12, np.median(np.abs(X))):
        rows_A = np.concatenate([np.arange(E, dtype=np.int64),
                                 np.arange(E, dtype=np.int64)])
        cols_A = np.concatenate([u, v])
        data_A = np.concatenate([alpha, alpha])
        # Augment with smoothness rows: sqrt(μ)·(X_u − X_v)
        sqrt_mu = np.sqrt(reg_mu)
        reg_rows = np.concatenate([np.arange(E, dtype=np.int64),
                                   np.arange(E, dtype=np.int64)])
        reg_cols = np.concatenate([u, v])
        reg_data = np.concatenate([np.full(E, sqrt_mu),
                                   np.full(E, -sqrt_mu)])
        rows_A = np.concatenate([rows_A, reg_rows + E])
        cols_A = np.concatenate([cols_A, reg_cols])
        data_A = np.concatenate([data_A, reg_data])
        A_sp = sp.coo_matrix((data_A, (rows_A, cols_A)),
                             shape=(2 * E, N)).tocsr()
        rhs = np.concatenate([w_star, np.zeros(E)])
        t1 = time.perf_counter()
        res = lsq_linear(A_sp, rhs, bounds=(0.0, np.inf),
                         method="trf", max_iter=200)
        dt = time.perf_counter() - t1
        X = res.x

    X = np.clip(X, 1e-12, None)
    return X, dt


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main(sigma: float = 0.10, reg_mu: float = 1e-4, seed: int = 42) -> None:
    t0 = time.perf_counter()

    # ---- 1. Load mesh ----
    obj_path = download_bunny()
    vertices, faces = load_obj(obj_path)
    edges = extract_edges(faces)
    vertices, faces, edges = largest_component(vertices, faces, edges)
    N, E, F = len(vertices), len(edges), len(faces)
    print(f"Loaded: {N:,} vertices, {F:,} faces, {E:,} edges.")

    # ---- 2. Ground-truth seam & conformal edge weights ----
    s_gt = seam_gt_3d(vertices)
    X_gt = np.exp(s_gt)
    u, v = edges[:, 0], edges[:, 1]
    l0 = np.linalg.norm(vertices[u] - vertices[v], axis=1)
    w_gt = l0 * (X_gt[u] + X_gt[v]) / 2.0

    # ---- 3. Add multiplicative noise ----
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal(E)
    w_star = np.clip(w_gt * (1.0 + sigma * xi), 1e-6, None)

    # ---- 4a. Baseline QP (Theorem 7, unregularised) ----
    print(f"Solving baseline QP (N={N:,}, E={E:,}, σ={sigma}) ...")
    t_solve = time.perf_counter()
    solver_stats: dict = {}
    X_base, _, _, _ = validation.solve_inverse_seam(
        vertices, edges, w_star, stats=solver_stats,
    )
    dt_base = time.perf_counter() - t_solve
    s_base = np.log(X_base)
    w_base = l0 * (X_base[u] + X_base[v]) / 2.0

    # ---- 4b. Laplacian-regularised QP ----
    print(f"Solving regularised QP (μ={reg_mu}) ...")
    X_reg, dt_reg = solve_inverse_seam_regularised(
        vertices, edges, w_star, reg_mu=reg_mu,
    )
    s_reg = np.log(X_reg)
    w_reg = l0 * (X_reg[u] + X_reg[v]) / 2.0

    # ---- 5. Metrics ----
    noisy_rel = float(np.linalg.norm(w_star - w_gt) / np.linalg.norm(w_gt))

    base_rel = float(np.linalg.norm(w_base - w_gt) / np.linalg.norm(w_gt))
    base_r = float(np.corrcoef(s_gt, s_base)[0, 1])
    a_base, b_base, r2_base = validation.compute_affine_fit(s_gt, s_base)

    reg_rel = float(np.linalg.norm(w_reg - w_gt) / np.linalg.norm(w_gt))
    reg_r = float(np.corrcoef(s_gt, s_reg)[0, 1])
    a_reg, b_reg, r2_reg = validation.compute_affine_fit(s_gt, s_reg)

    # Triangle-inequality audit
    tri_viol_gt, _ = triangle_inequality_check(faces, edges, w_gt)
    tri_viol_base, _ = triangle_inequality_check(faces, edges, w_base)
    tri_viol_reg, _ = triangle_inequality_check(faces, edges, w_reg)

    print(f"\n{'=' * 70}")
    print("  Stanford Bunny — Inverse Seam Design Results")
    print(f"{'=' * 70}")
    print(f"  Mesh size         {N:>8,} verts   {E:>8,} edges   {F:>8,} faces")
    print(f"  Noise level σ              {sigma:.2f}")
    print(f"  Regularisation μ           {reg_mu:.1e}")
    print(f"{'─' * 70}")
    print(f"  {'':30s} {'Baseline':>12s}  {'Regularised':>12s}")
    print(f"  {'Solve time':30s} {dt_base:>11.2f}s  {dt_reg:>11.2f}s")
    print(f"  {'Edge-weight error (noisy)':30s} {noisy_rel:>12.6f}  {noisy_rel:>12.6f}")
    print(f"  {'Edge-weight error (recov.)':30s} {base_rel:>12.6f}  {reg_rel:>12.6f}")
    print(f"  {'Error reduction factor':30s} {noisy_rel/(base_rel+1e-15):>11.1f}×  {noisy_rel/(reg_rel+1e-15):>11.1f}×")
    print(f"  {'Seam Pearson r':30s} {base_r:>12.6f}  {reg_r:>12.6f}")
    print(f"  {'Seam R²':30s} {r2_base:>12.6f}  {r2_reg:>12.6f}")
    print(f"  {'Tri-ineq violations':30s} {tri_viol_base:>8d}/{F}  {tri_viol_reg:>8d}/{F}")
    print(f"{'=' * 70}\n")

    # ---- 6. Publication figures ----
    print("Rendering figures ...")

    # Per-edge relative errors
    rel_noisy = np.abs(w_star - w_gt) / (np.abs(w_gt) + 1e-12)
    rel_base = np.abs(w_base - w_gt) / (np.abs(w_gt) + 1e-12)
    rel_reg = np.abs(w_reg - w_gt) / (np.abs(w_gt) + 1e-12)
    fe0, fe1, fe2 = face_edge_indices(faces, edges)
    rel_gt_face = np.zeros(F)
    rel_noisy_face = (rel_noisy[fe0] + rel_noisy[fe1] + rel_noisy[fe2]) / 3.0
    rel_base_face = (rel_base[fe0] + rel_base[fe1] + rel_base[fe2]) / 3.0
    rel_reg_face = (rel_reg[fe0] + rel_reg[fe1] + rel_reg[fe2]) / 3.0
    rel_cap = 0.16
    err_norm = Normalize(vmin=0.0, vmax=rel_cap)
    sm_err = plt.cm.ScalarMappable(cmap=plt.colormaps["RdYlGn_r"], norm=err_norm)

    # --- Figure A: 4 bunny panels + colorbar ---
    fig_mesh = plt.figure(figsize=(16, 5.2))
    pw = 0.21    # panel width (fraction of figure)
    gap = 0.008  # gap between panels
    x0 = 0.005

    # Panel 1 — ground truth (zero error)
    ax1 = fig_mesh.add_axes([x0, 0.05, pw, 0.85])
    render_mesh(
        ax1, vertices, faces, rel_gt_face,
        r"Ground truth (error $\equiv$ 0)",
        cmap_name="RdYlGn_r", vmin=0.0, vmax=rel_cap, per_face=True,
    )

    # Panel 2 — noisy edge-weight error
    ax2 = fig_mesh.add_axes([x0 + pw + gap, 0.05, pw, 0.85])
    render_mesh(
        ax2, vertices, faces, rel_noisy_face,
        rf"Noisy  (med.={np.median(rel_noisy):.3f})",
        cmap_name="RdYlGn_r", vmin=0.0, vmax=rel_cap, per_face=True,
    )

    # Panel 3 — baseline (unregularised) recovered error
    ax3 = fig_mesh.add_axes([x0 + 2*(pw + gap), 0.05, pw, 0.85])
    render_mesh(
        ax3, vertices, faces, rel_base_face,
        rf"Baseline QP  (med.={np.median(rel_base):.3f})",
        cmap_name="RdYlGn_r", vmin=0.0, vmax=rel_cap, per_face=True,
    )

    # Panel 4 — regularised recovered error
    ax4 = fig_mesh.add_axes([x0 + 3*(pw + gap), 0.05, pw, 0.85])
    render_mesh(
        ax4, vertices, faces, rel_reg_face,
        rf"Regularised QP  (med.={np.median(rel_reg):.3f})",
        cmap_name="RdYlGn_r", vmin=0.0, vmax=rel_cap, per_face=True,
    )

    # Shared colorbar
    cbar_x = x0 + 4*(pw + gap)
    cax_err = fig_mesh.add_axes([cbar_x, 0.15, 0.008, 0.60])
    cb_err = fig_mesh.colorbar(sm_err, cax=cax_err)
    cb_err.set_label("relative metric error", fontsize=10)

    _save(fig_mesh, "fig3_bunny_mesh")
    plt.close(fig_mesh)

    # --- Figure B: edge-weight error histogram ---
    fig_hist, ax5 = plt.subplots(figsize=(6, 4))
    lo_bin = float(min(
        max(np.min(rel_noisy), 1e-8),
        max(np.min(rel_base), 1e-8),
        max(np.min(rel_reg), 1e-8),
    ))
    hi_bin = float(np.percentile(rel_noisy, 99.5))
    bins = np.logspace(np.log10(lo_bin), np.log10(hi_bin), 50)
    ax5.hist(rel_noisy, bins=bins, alpha=0.55, label=r"noisy $w^*$", color="#d62728")
    ax5.hist(rel_base, bins=bins, alpha=0.55, label=r"baseline $w_{\mathrm{opt}}$", color="#ff7f0e")
    ax5.hist(rel_reg, bins=bins, alpha=0.55, label=r"regularized $w_{\mathrm{opt}}$", color="#2ca02c")
    ax5.set_xscale("log")
    ax5.set_xlabel(r"$|w - w_{\mathrm{gt}}| \;/\; |w_{\mathrm{gt}}|$", fontsize=10)
    ax5.set_ylabel("count", fontsize=10)
    ax5.set_title("Edge-weight error distribution", fontsize=11)
    ax5.legend(fontsize=9, frameon=False)
    ax5.grid(True, alpha=0.25)
    fig_hist.tight_layout()

    _save(fig_hist, "fig4_bunny_histogram")
    plt.close(fig_hist)

    dt_total = time.perf_counter() - t0
    print(
        f"Saved to {FIG_DIR / 'fig3_bunny_mesh'}.{{pdf,png}} and "
        f"{FIG_DIR / 'fig4_bunny_histogram'}.{{pdf,png}}  "
        f"(total {dt_total:.1f} s)"
    )


if __name__ == "__main__":
    main()
