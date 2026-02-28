"""
Benchmark: Seam QP vs industry-standard baselines on the Stanford Bunny.

Compares six methods for recovering clean edge weights from noisy targets:

  1. Noisy targets w* (no processing)
  2. Laplacian edge-weight smoothing (graph diffusion, fast)
  3. Triangle-inequality-constrained QP (metric projection via OSQP)
  4. Seam QP — baseline (Theorem 7, sparse direct solve)
  5. Seam QP — Laplacian-regularised (Corollary, sparse direct solve)
  6. Seam QP — Laplacian-regularised via OSQP (same problem, ADMM solver)

Metrics: relative weight error, triangle-inequality violations, wall time.
"""

from __future__ import annotations

import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import osqp

import bunny
import validation


def laplacian_edge_smooth(
    edges: np.ndarray, faces: np.ndarray, w: np.ndarray,
    n_iter: int = 2, mix: float = 0.05,
) -> tuple[np.ndarray, float]:
    """Smooth edge weights by averaging over adjacent edges in each face.

    For each face (a,b,c), each edge receives a contribution from its two
    face-mates.  This is the edge-graph analogue of Laplacian smoothing.

    Parameters are **oracle-tuned** against the ground truth to give this
    baseline its best possible result (n_iter=2, mix=0.05 minimises the
    error over a grid sweep).

    Returns (w_smooth, time).
    """
    E = len(edges)

    # Face edge indices (precomputed outside timer for consistency)
    fe0, fe1, fe2 = bunny.face_edge_indices(faces, edges)

    t0 = time.perf_counter()
    ws = w.copy()
    for _ in range(n_iter):
        w_new = np.zeros(E)
        count = np.zeros(E)
        for ea, eb, ec in [(fe0, fe1, fe2), (fe1, fe2, fe0), (fe2, fe0, fe1)]:
            # Each edge gets contributions from its two face-mates
            w_new[ea] += ws[eb] + ws[ec]
            count[ea] += 2
        count = np.maximum(count, 1)
        ws = (1.0 - mix) * ws + mix * (w_new / count)
        ws = np.clip(ws, 1e-12, None)
    dt = time.perf_counter() - t0
    return ws, dt


def triangle_ineq_projection_osqp(
    faces: np.ndarray, edges: np.ndarray, w_star: np.ndarray
) -> tuple[np.ndarray, float]:
    """Project onto the metric cone: min ||w - w*||² s.t. triangle inequalities.

    For each face with edges (a, b, c):
      w_a + w_b >= w_c   =>   w_a + w_b - w_c >= 0
      w_b + w_c >= w_a   =>  -w_a + w_b + w_c >= 0
      w_c + w_a >= w_b   =>   w_a - w_b + w_c >= 0
    Also: w >= 0.

    Solved with OSQP (ADMM-based first-order QP solver).
    """
    E = len(edges)
    F = len(faces)

    # Objective: ½ ||w - w*||² = ½ w^T I w - w*^T w + const
    P = sp.eye(E, format="csc")
    q = -w_star.copy()

    # Build triangle-inequality constraint matrix
    # 3 constraints per face + E positivity constraints
    fe0, fe1, fe2 = bunny.face_edge_indices(faces, edges)

    rows = []
    cols = []
    data = []
    for ci, (ea, eb, ec) in enumerate(
        [(fe0, fe1, fe2), (fe1, fe2, fe0), (fe2, fe0, fe1)]
    ):
        # Constraint: w[ea] + w[eb] - w[ec] >= 0
        base_row = ci * F
        row_idx = np.arange(F) + base_row
        # +1 for ea
        rows.append(row_idx)
        cols.append(ea)
        data.append(np.ones(F))
        # +1 for eb
        rows.append(row_idx)
        cols.append(eb)
        data.append(np.ones(F))
        # -1 for ec
        rows.append(row_idx)
        cols.append(ec)
        data.append(-np.ones(F))

    # Also add positivity: w >= 0 (identity rows)
    pos_base = 3 * F
    pos_rows = np.arange(E) + pos_base
    rows.append(pos_rows)
    cols.append(np.arange(E))
    data.append(np.ones(E))

    n_constraints = 3 * F + E
    A = sp.csc_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_constraints, E),
    )
    l = np.zeros(n_constraints)  # lower bound: all >= 0
    u = np.full(n_constraints, np.inf)  # no upper bound

    solver = osqp.OSQP()
    t0 = time.perf_counter()
    solver.setup(
        P, q, A, l, u,
        verbose=False,
        eps_abs=1e-9,
        eps_rel=1e-9,
        max_iter=100000,
        polish=True,
    )
    result = solver.solve()
    dt = time.perf_counter() - t0  # includes setup + solve

    w_proj = np.clip(result.x, 1e-12, None)
    return w_proj, dt


def seam_qp_osqp(
    vertices: np.ndarray,
    edges: np.ndarray,
    w_star: np.ndarray,
    reg_mu: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """Solve the Laplacian-regularised seam QP via OSQP (ADMM solver).

    Same problem as solve_inverse_seam_regularised but using OSQP.
    min  ½ X^T (H + μL) X − b^T X   s.t.  X ≥ 0.
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

    P = (H + reg_mu * L.tocsr()).tocsc()

    # Positivity constraints: X >= 0
    A_constr = sp.eye(N, format="csc")
    l_constr = np.zeros(N)
    u_constr = np.full(N, np.inf)

    solver = osqp.OSQP()
    t0 = time.perf_counter()
    solver.setup(
        P, -b, A_constr, l_constr, u_constr,
        verbose=False,
        eps_abs=1e-9,
        eps_rel=1e-9,
        max_iter=100000,
        polish=True,
    )
    result = solver.solve()
    dt = time.perf_counter() - t0  # includes setup + solve

    X = np.clip(result.x, 1e-12, None)
    return X, dt


def main(sigma: float = 0.10, reg_mu: float = 1e-4, seed: int = 42) -> None:
    # ---- Load mesh ----
    obj_path = bunny.download_bunny()
    vertices, faces = bunny.load_obj(obj_path)
    edges = bunny.extract_edges(faces)
    vertices, faces, edges = bunny.largest_component(vertices, faces, edges)
    N, E, F = len(vertices), len(edges), len(faces)
    print(f"Mesh: {N:,} verts, {E:,} edges, {F:,} faces\n")

    # ---- Ground-truth ----
    s_gt = bunny.seam_gt_3d(vertices)
    X_gt = np.exp(s_gt)
    u, v = edges[:, 0], edges[:, 1]
    l0 = np.linalg.norm(vertices[u] - vertices[v], axis=1)
    w_gt = l0 * (X_gt[u] + X_gt[v]) / 2.0

    # ---- Noisy targets ----
    rng = np.random.default_rng(seed)
    xi = rng.standard_normal(E)
    w_star = np.clip(w_gt * (1.0 + sigma * xi), 1e-6, None)

    # ---- Method 1: Noisy (no processing) ----
    print("Method 1: Noisy targets (no processing)")
    w_noisy = w_star.copy()

    # ---- Method 2: Laplacian edge smoothing (oracle-tuned) ----
    print("Method 2: Laplacian edge smoothing (oracle-tuned) ...")
    w_lap, dt_lap = laplacian_edge_smooth(edges, faces, w_star)  # n_iter=2, mix=0.05

    # ---- Method 3: Triangle-inequality metric projection (OSQP) ----
    print("Method 3: Triangle-inequality projection (OSQP) ...")
    w_tri, dt_tri = triangle_ineq_projection_osqp(faces, edges, w_star)

    # ---- Method 4: Seam QP baseline (spsolve) ----
    print("Method 4: Seam QP baseline (spsolve) ...")
    t0 = time.perf_counter()
    X_base, _, _, _ = validation.solve_inverse_seam(
        vertices, edges, w_star, stats={},
    )
    dt_base = time.perf_counter() - t0
    w_base = l0 * (X_base[u] + X_base[v]) / 2.0

    # ---- Method 5: Seam QP regularised (spsolve) ----
    print("Method 5: Seam QP regularised (spsolve) ...")
    X_reg, dt_reg = bunny.solve_inverse_seam_regularised(
        vertices, edges, w_star, reg_mu=reg_mu,
    )
    w_reg = l0 * (X_reg[u] + X_reg[v]) / 2.0

    # ---- Method 6: Seam QP regularised via OSQP ----
    print("Method 6: Seam QP regularised (OSQP) ...")
    X_osqp, dt_osqp = seam_qp_osqp(vertices, edges, w_star, reg_mu=reg_mu)
    w_osqp = l0 * (X_osqp[u] + X_osqp[v]) / 2.0

    # ---- Evaluate all methods ----
    methods = [
        ("Noisy (no processing)", w_noisy, 0.0),
        ("Laplacian smoothing", w_lap, dt_lap),
        ("Tri-ineq projection (OSQP)", w_tri, dt_tri),
        ("Seam QP baseline (spsolve)", w_base, dt_base),
        ("Seam QP reg. (spsolve)", w_reg, dt_reg),
        ("Seam QP reg. (OSQP)", w_osqp, dt_osqp),
    ]

    print(f"\n{'=' * 100}")
    print(f"  BENCHMARK: Stanford Bunny ({N:,} V, {E:,} E, {F:,} F)  σ={sigma}  μ={reg_mu}")
    print(f"{'=' * 100}")
    header = (
        f"  {'Method':<32s}  {'w_err':>8s}  {'reduction':>10s}  "
        f"{'tri-viol':>12s}  {'time':>8s}"
    )
    print(header)
    print(f"  {'─' * 94}")

    noisy_err = float(np.linalg.norm(w_star - w_gt) / np.linalg.norm(w_gt))

    for name, w_method, dt in methods:
        rel_err = float(np.linalg.norm(w_method - w_gt) / np.linalg.norm(w_gt))
        reduction = noisy_err / max(rel_err, 1e-15)
        tri_viol, _ = bunny.triangle_inequality_check(faces, edges, w_method)
        time_str = f"{dt:.3f}s" if dt > 0 else "—"
        print(
            f"  {name:<32s}  {rel_err:>8.4f}  {reduction:>9.1f}×  "
            f"{tri_viol:>6d}/{F}  {time_str:>8s}"
        )

    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
