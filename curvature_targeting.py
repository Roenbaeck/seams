"""
Curvature-Targeting Experiment via the Seam Newton Step (Proposition 13)

Manufactured-solution test on the Stanford Bunny:
  1. Choose a known smooth seam  s_true  (height-weighted, small amplitude).
  2. Compute the target curvature  K* = K(s_true)  exactly.
  3. Single Newton step from s = 0:  verify  ||K_s - K*|| = O(||s||^2).
  4. Iterated Newton:  converge to  s_true  up to gauge.
  5. Scaling test:  alpha * s_true,  verify  nonlinear residual ~ alpha^2.

Dependencies: numpy, scipy, matplotlib (same as bunny.py).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from bunny import (
    download_bunny,
    load_obj,
    extract_edges,
    largest_component,
    _save,
    FIG_DIR,
)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _angles_from_lengths(a, b, c):
    """Angles opposite sides a, b, c (vectorized, clamped)."""
    cos_A = np.clip((b**2 + c**2 - a**2) / (2 * b * c + 1e-30), -1, 1)
    cos_B = np.clip((a**2 + c**2 - b**2) / (2 * a * c + 1e-30), -1, 1)
    cos_C = np.clip((a**2 + b**2 - c**2) / (2 * a * b + 1e-30), -1, 1)
    return np.arccos(cos_A), np.arccos(cos_B), np.arccos(cos_C)


def edge_lengths_from_vertices(vertices, edges):
    return np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)


def seam_edge_lengths(ell0, edges, s):
    """ell_s(u,v) = ell_0(u,v) * (e^{s(u)} + e^{s(v)}) / 2."""
    X = np.exp(s)
    return ell0 * 0.5 * (X[edges[:, 0]] + X[edges[:, 1]])


def build_edge_index(edges):
    idx = {}
    for k, (u, v) in enumerate(edges):
        idx[(int(u), int(v))] = k
    return idx


def face_edge_weights(faces, edge_idx, w):
    """Per-face edge lengths: (a=opp v0, b=opp v1, c=opp v2)."""
    F = len(faces)
    wa, wb, wc = np.empty(F), np.empty(F), np.empty(F)
    for fi in range(F):
        i, j, k = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        wa[fi] = w[edge_idx[(min(j, k), max(j, k))]]
        wb[fi] = w[edge_idx[(min(k, i), max(k, i))]]
        wc[fi] = w[edge_idx[(min(i, j), max(i, j))]]
    return wa, wb, wc


def find_boundary_vertices(N, faces):
    """Boundary = vertices incident to a half-edge with no twin."""
    hec: dict[tuple[int, int], int] = {}
    for fi in range(len(faces)):
        i, j, k = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        for u, v in [(i, j), (j, k), (k, i)]:
            hec[(u, v)] = hec.get((u, v), 0) + 1
    bdy = set()
    for (u, v) in hec:
        if (v, u) not in hec:
            bdy.add(u); bdy.add(v)
    return np.array(sorted(bdy), dtype=np.int64)


def angle_defect_curvature(N, faces, wa, wb, wc, bdy_verts=None):
    """K(u) = 2*pi - sum(theta)  (interior) or  pi - sum(theta)  (boundary)."""
    A, B, C = _angles_from_lengths(wa, wb, wc)
    asum = np.zeros(N)
    np.add.at(asum, faces[:, 0], A)
    np.add.at(asum, faces[:, 1], B)
    np.add.at(asum, faces[:, 2], C)
    K = 2.0 * np.pi - asum
    if bdy_verts is not None and len(bdy_verts) > 0:
        K[bdy_verts] = np.pi - asum[bdy_verts]
    return K


def cotangent_laplacian(N, faces, edges, edge_idx, wa, wb, wc):
    """Cotangent Laplacian  L^cot  for the given edge-length metric."""
    A, B, C = _angles_from_lengths(wa, wb, wc)
    cot_A = np.cos(A) / (np.sin(A) + 1e-30)
    cot_B = np.cos(B) / (np.sin(B) + 1e-30)
    cot_C = np.cos(C) / (np.sin(C) + 1e-30)
    E = len(edges)
    cw = np.zeros(E)
    for fi in range(len(faces)):
        i, j, k = int(faces[fi, 0]), int(faces[fi, 1]), int(faces[fi, 2])
        cw[edge_idx[(min(j, k), max(j, k))]] += cot_A[fi]
        cw[edge_idx[(min(k, i), max(k, i))]] += cot_B[fi]
        cw[edge_idx[(min(i, j), max(i, j))]] += cot_C[fi]
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([-0.5 * cw, -0.5 * cw])
    L = sp.coo_matrix((data, (row, col)), shape=(N, N)).tocsr()
    diag = -np.array(L.sum(axis=1)).ravel()
    L = L + sp.diags(diag)
    return L


# ---------------------------------------------------------------------------
# Newton solver
# ---------------------------------------------------------------------------

def curvature_newton(
    vertices, faces, edges, K_target, bdy_verts,
    n_iter=20, damping=1.0, verbose=True,
):
    """Iterated seam Newton steps (Proposition 13)."""
    N = len(vertices)
    ell0 = edge_lengths_from_vertices(vertices, edges)
    edge_idx = build_edge_index(edges)
    s = np.zeros(N)
    history = []

    for it in range(n_iter):
        w = seam_edge_lengths(ell0, edges, s)
        wa, wb, wc = face_edge_weights(faces, edge_idx, w)
        K_s = angle_defect_curvature(N, faces, wa, wb, wc, bdy_verts)

        rhs = K_target - K_s
        res_l2 = float(np.linalg.norm(rhs))
        res_linf = float(np.max(np.abs(rhs)))
        rec = {"iter": it, "residual_l2": res_l2, "residual_linf": res_linf,
               "seam_linf": float(np.max(np.abs(s)))}
        history.append(rec)
        if verbose:
            print(f"  iter {it:2d}:  ||r||_2={res_l2:.4e}  "
                  f"||r||_inf={res_linf:.4e}  ||s||_inf={rec['seam_linf']:.4e}")
        if res_l2 < 1e-10:
            if verbose:
                print("  Converged.")
            break

        L = cotangent_laplacian(N, faces, edges, edge_idx, wa, wb, wc)
        rhs_proj = rhs - rhs.mean()
        ds = spsolve(L + sp.eye(N) * 1e-8, rhs_proj)
        ds -= ds.mean()

        # Line search: keep ||s||_inf bounded
        alpha = damping
        for _ in range(15):
            sc = s + alpha * ds
            sc -= sc.mean()
            if np.max(np.abs(sc)) < 4.0:
                break
            alpha *= 0.5

        s = s + alpha * ds
        s -= s.mean()

    return s, history


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    t0 = time.perf_counter()

    # --- Load mesh ---------------------------------------------------------
    download_bunny()
    vertices, faces = load_obj(Path(__file__).resolve().parent / "stanford-bunny.obj")
    edges = extract_edges(faces)
    vertices, faces, edges = largest_component(vertices, faces, edges)
    N, E, F = len(vertices), len(edges), len(faces)
    print(f"Mesh: {N:,} vertices, {E:,} edges, {F:,} faces")

    bdy = find_boundary_vertices(N, faces)
    chi = N - E + F
    print(f"Boundary vertices: {len(bdy)}  |  chi = {chi}")

    # --- Background geometry -----------------------------------------------
    ell0 = edge_lengths_from_vertices(vertices, edges)
    edge_idx = build_edge_index(edges)
    wa0, wb0, wc0 = face_edge_weights(faces, edge_idx, ell0)
    K_0 = angle_defect_curvature(N, faces, wa0, wb0, wc0, bdy)
    print(f"sum(K_0) = {K_0.sum():.4f}  (2*pi*chi = {2*np.pi*chi:.4f})")

    # --- Manufactured ground truth -----------------------------------------
    # Smooth seam: s_true(v) = A * sin(pi * normalized_height)
    # Small amplitude so seam stays in the linearisation regime.
    y = vertices[:, 1]
    yn = (y - y.min()) / (y.max() - y.min() + 1e-30)
    amplitude = 0.15
    s_true = amplitude * np.sin(np.pi * yn)
    s_true -= s_true.mean()                           # gauge: mean = 0
    s_true[bdy] = 0.0                                 # pin boundary
    print(f"\nManufactured seam:  A = {amplitude},  ||s_true||_inf = {np.max(np.abs(s_true)):.4f}")

    # Compute K* = K(s_true) exactly
    w_true = seam_edge_lengths(ell0, edges, s_true)
    wa_t, wb_t, wc_t = face_edge_weights(faces, edge_idx, w_true)
    K_star = angle_defect_curvature(N, faces, wa_t, wb_t, wc_t, bdy)
    dK = K_star - K_0
    print(f"||K* - K_0||_2  = {np.linalg.norm(dK):.6f}")
    print(f"||K* - K_0||_inf = {np.max(np.abs(dK)):.6f}")
    print(f"sum(K*)         = {K_star.sum():.4f}")

    # --- Build L^cot at s = 0 ---------------------------------------------
    L0 = cotangent_laplacian(N, faces, edges, edge_idx, wa0, wb0, wc0)

    # ===================================================================
    # Part A: Proposition 13 verification  --  O(||s||^2) scaling
    # ===================================================================
    print("\n=== Part A: Proposition 13 -- O(||s||^2) scaling ===")
    print(f"  {'alpha':>6}  {'||s||_inf':>10}  {'||nonlin err||':>14}  {'ratio/||s||^2':>14}")
    alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    for alpha in alphas:
        sa = alpha * s_true
        wa_a = seam_edge_lengths(ell0, edges, sa)
        wa_a, wb_a, wc_a = face_edge_weights(faces, edge_idx, wa_a)
        Ka = angle_defect_curvature(N, faces, wa_a, wb_a, wc_a, bdy)
        K_lin = K_0 + L0 @ sa                         # linear prediction
        nlerr = np.linalg.norm(Ka - K_lin)
        sn2 = np.linalg.norm(sa)**2
        ratio = nlerr / (sn2 + 1e-30)
        print(f"  {alpha:6.2f}  {np.max(np.abs(sa)):10.6f}  "
              f"{nlerr:14.6e}  {ratio:14.6f}")

    # ===================================================================
    # Part B: Single Newton step
    # ===================================================================
    print("\n=== Part B: Single Newton step ===")
    rhs0 = dK - dK.mean()
    s_single = spsolve(L0 + sp.eye(N) * 1e-8, rhs0)
    s_single -= s_single.mean()

    ws = seam_edge_lengths(ell0, edges, s_single)
    wa_s, wb_s, wc_s = face_edge_weights(faces, edge_idx, ws)
    K_single = angle_defect_curvature(N, faces, wa_s, wb_s, wc_s, bdy)

    err_s_l2 = np.linalg.norm(K_single - K_star)
    err_s_linf = np.max(np.abs(K_single - K_star))
    print(f"  ||s_est||_inf      = {np.max(np.abs(s_single)):.6f}")
    print(f"  ||K_s - K*||_2     = {err_s_l2:.6e}")
    print(f"  ||K_s - K*||_inf   = {err_s_linf:.6e}")
    # Compare seams (up to gauge)
    ds_gauge = s_single - s_true
    ds_gauge -= ds_gauge.mean()
    print(f"  ||s_est - s_true||_2  = {np.linalg.norm(ds_gauge):.6e}")
    print(f"  ||s_est - s_true||_inf = {np.max(np.abs(ds_gauge)):.6e}")

    # ===================================================================
    # Part C: Iterated Newton  -->  converges to s_true
    # ===================================================================
    print("\n=== Part C: Iterated Newton ===")
    s_iter, history = curvature_newton(
        vertices, faces, edges, K_star, bdy,
        n_iter=20, damping=1.0, verbose=True,
    )

    wi = seam_edge_lengths(ell0, edges, s_iter)
    wa_i, wb_i, wc_i = face_edge_weights(faces, edge_idx, wi)
    K_iter = angle_defect_curvature(N, faces, wa_i, wb_i, wc_i, bdy)

    ds_iter = s_iter - s_true; ds_iter -= ds_iter.mean()
    print(f"\n  Final ||K_s - K*||_2    = {np.linalg.norm(K_iter - K_star):.6e}")
    print(f"  Final ||s - s_true||_inf = {np.max(np.abs(ds_iter)):.6e}")

    # ===================================================================
    # Figures
    # ===================================================================
    print("\nGenerating figures ...")

    # --- Convergence plot ---
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.8))
    iters  = [h["iter"] for h in history]
    resids = [h["residual_l2"] for h in history]
    ax2.semilogy(iters, resids, "o-", color="#1f77b4", lw=1.5, ms=5)
    ax2.set_xlabel("Newton iteration", fontsize=11)
    ax2.set_ylabel(r"$\|K_s - K^*\|_2$", fontsize=11)
    ax2.set_title("Curvature-targeting convergence", fontsize=12)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    _save(fig2, "fig6_curvature_convergence")
    plt.close(fig2)

    # Panel 3: O(||s||^2) log-log
    fig3, ax3 = plt.subplots(figsize=(5.5, 3.8))
    snorms = []
    nlerrs = []
    for alpha in alphas:
        sa = alpha * s_true
        wa_a2 = seam_edge_lengths(ell0, edges, sa)
        wa_a2, wb_a2, wc_a2 = face_edge_weights(faces, edge_idx, wa_a2)
        Ka2 = angle_defect_curvature(N, faces, wa_a2, wb_a2, wc_a2, bdy)
        K_lin2 = K_0 + L0 @ sa
        snorms.append(np.linalg.norm(sa))
        nlerrs.append(np.linalg.norm(Ka2 - K_lin2))
    ax3.loglog(snorms, nlerrs, "s-", color="#d62728", lw=1.5, ms=6, label="nonlinear residual")
    # Reference O(||s||^2) line
    sref = np.array(snorms)
    c = nlerrs[-1] / sref[-1]**2
    ax3.loglog(sref, c * sref**2, "--", color="gray", lw=1, label=r"$O(\|s\|^2)$ reference")
    ax3.set_xlabel(r"$\|s\|_2$", fontsize=11)
    ax3.set_ylabel(r"$\|K_s - (K_0 + L^{\mathrm{cot}} s)\|_2$", fontsize=11)
    ax3.set_title(r"Proposition 13: $O(\|s\|^2)$ accuracy", fontsize=12)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    _save(fig3, "fig7_quadratic_scaling")
    plt.close(fig3)

    dt = time.perf_counter() - t0
    print(f"\nDone in {dt:.1f} s.  Figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()
