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
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

def generate_mesh(n_points=500):
    """Generates a random 2D triangulated mesh (non-bipartite graph)."""
    np.random.seed(42)
    # Generate random points in the unit square
    points = np.random.rand(n_points, 2)
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

def main():
    print("--- Seam-Driven Geometry Validation ---")
    
    # 1. Setup Background Mesh & Metric
    points, edges, faces = generate_mesh(n_points=800)
    N = len(points)
    E = len(edges)

    u = edges[:, 0].astype(np.int64)
    v = edges[:, 1].astype(np.int64)
    
    # Background lengths l_0 (Euclidean distance)
    pts_u, pts_v = points[u], points[v]
    l0 = np.linalg.norm(pts_u - pts_v, axis=1)
    
    # 2. Define Ground Truth Seam (s_gt)
    # A smooth scalar field over the mesh: s(x,y) = sin(2*pi*x) * cos(2*pi*y)
    s_gt = np.sin(2 * np.pi * points[:, 0]) * np.cos(2 * np.pi * points[:, 1])
    
    # 3. Generate Target Weights (Conformal Rule + Noise)
    # True metric weights governed by the Conformal Graph Rule
    X_gt = np.exp(s_gt)
    w_gt = l0 * (X_gt[u] + X_gt[v]) / 2.0
    
    # Add heavy 15% Gaussian noise to make the target weights physically invalid
    noise_level = 0.15
    noise = noise_level * np.random.randn(E)
    w_star = w_gt * (1.0 + noise)
    
    # Ensure strict positivity (lengths must be > 0)
    w_star = np.clip(w_star, 1e-6, None)
    
    print(f"Mesh generated: {N} vertices, {E} edges.")
    print(f"Added {noise_level*100}% Gaussian noise to ground-truth metric.")

    # 4. Construct the Strictly Convex QP (Theorem 7)
    # Energy: E(X) = sum_e ( (l0_e / 2) * (X_u + X_v) - w*_e )^2
    # Setting grad(E) = 0 yields the linear system: H * X = b
    # where H is the Signless Laplacian weighted by (l0_e / 2)^2
    
    alpha = l0 / 2.0
    alpha_sq = alpha**2

    # Hessian H is proportional to the weighted signless Laplacian (Theorem thm:inverse_seam):
    #   H_uu = sum_{v~u} (l0(u,v)^2 / 4) = sum alpha_e^2
    #   H_uv = alpha_e^2 for edges (u,v)
    I = np.concatenate([u, v])
    J = np.concatenate([v, u])
    V_off = np.concatenate([alpha_sq, alpha_sq])

    H_off = sp.coo_matrix((V_off, (I, J)), shape=(N, N))
    H_diag_vals = np.bincount(u, weights=alpha_sq, minlength=N) + np.bincount(v, weights=alpha_sq, minlength=N)
    H = (H_off + sp.diags(H_diag_vals)).tocsr()

    # Linear term b = A^T w*, where A has alpha_e at columns u and v.
    b_edge = alpha * w_star
    b = np.bincount(u, weights=b_edge, minlength=N) + np.bincount(v, weights=b_edge, minlength=N)

    # 5. Solve the System (O(E) sparse solve)
    print("\nSolving Signless Laplacian system H X = b...")
    # Because a triangulated mesh is non-bipartite, H is strictly positive definite.
    # The unconstrained minimizer solves HX=b. If it violates X>0, fall back to a
    # positivity-constrained least squares solve for robustness.
    X_opt = spsolve(H, b)

    if not np.all(np.isfinite(X_opt)) or np.min(X_opt) <= 0:
        print("Unconstrained solution has non-positive entries; solving with X>=0 constraints...")
        rows = np.repeat(np.arange(E, dtype=np.int64), 2)
        cols = np.concatenate([u, v])
        data = np.concatenate([alpha, alpha])
        A = sp.coo_matrix((data, (rows, cols)), shape=(E, N)).tocsr()
        res = lsq_linear(A, w_star, bounds=(0.0, np.inf), method="trf")
        if not res.success:
            raise RuntimeError(f"lsq_linear failed: {res.message}")
        X_opt = res.x
    
    # Recover the scalar seam
    X_opt = np.clip(X_opt, 1e-12, None) # Guard against numerical precision negatives
    s_opt = np.log(X_opt)
    
    # 6. Evaluate Results
    w_opt = l0 * (X_opt[u] + X_opt[v]) / 2.0

    # Baselines / extra diagnostics
    # "Do nothing" seam: X=1 => w=l0
    w_base = l0
    # Noise-only deviation (how far the noisy target is from the realizable ground truth)
    err_noise_weights = np.linalg.norm(w_gt - w_star) / np.linalg.norm(w_star)
    err_noise_to_truth_weights = np.linalg.norm(w_star - w_gt) / np.linalg.norm(w_gt)
    # Denoising quality (how close the recovered weights are to the ground truth)
    err_denoise_weights = np.linalg.norm(w_opt - w_gt) / np.linalg.norm(w_gt)
    err_base_weights = np.linalg.norm(w_base - w_star) / np.linalg.norm(w_star)
    err_X = np.linalg.norm(X_opt - X_gt) / np.linalg.norm(X_gt)
    pearson_r = float(np.corrcoef(s_gt, s_opt)[0, 1])

    # Best-fit affine map s_opt ≈ a*s_gt + b
    s_gt_mean = float(np.mean(s_gt))
    s_opt_mean = float(np.mean(s_opt))
    s_gt_centered = s_gt - s_gt_mean
    denom = float(np.dot(s_gt_centered, s_gt_centered))
    if denom > 0:
        a_fit = float(np.dot(s_gt_centered, s_opt - s_opt_mean) / denom)
    else:
        a_fit = float("nan")
    b_fit = s_opt_mean - a_fit * s_gt_mean
    residual = s_opt - (a_fit * s_gt + b_fit)
    ss_res = float(np.dot(residual, residual))
    ss_tot = float(np.dot(s_opt - s_opt_mean, s_opt - s_opt_mean))
    r2_fit = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    
    # Compute relative L2 Errors
    err_weights = np.linalg.norm(w_opt - w_star) / np.linalg.norm(w_star)
    err_seam = np.linalg.norm(s_opt - s_gt) / np.linalg.norm(s_gt)
    
    print("\n--- Results ---")
    print(f"Target fitting error (||w_opt - w*||_2) : {err_weights:.4f}")
    print(f"Ground Truth Seam recovery error        : {err_seam:.4f}")

    print("\n--- Extra diagnostics ---")
    print(f"Noise-only weight deviation (||w_gt - w*||_2 / ||w*||_2) : {err_noise_weights:.4f}")
    print(f"Noisy-to-truth weight error   (||w*   - w_gt||_2 / ||w_gt||_2): {err_noise_to_truth_weights:.4f}")
    print(f"Denoised-to-truth weight error (||w_opt - w_gt||_2 / ||w_gt||_2): {err_denoise_weights:.4f}")
    print(f"Baseline X=1 weight error  (||l0  - w*||_2 / ||w*||_2)   : {err_base_weights:.4f}")
    if np.isfinite(err_base_weights) and err_base_weights > 0:
        improvement = 1.0 - (err_weights / err_base_weights)
        print(f"Improvement vs X=1 baseline                               : {improvement*100:.1f}%")
    print(f"Relative X recovery error (||X_opt - X_gt||_2 / ||X_gt||_2): {err_X:.4f}")
    print(f"Pearson corr(s_gt, s_opt)                                  : {pearson_r:.4f}")
    print(f"Affine fit: s_opt ≈ a*s_gt + b (a, b)                      : ({a_fit:.4f}, {b_fit:.4f})")
    print(f"Affine fit R^2 (predicting s_opt from s_gt)                : {r2_fit:.4f}")
    
    # 7. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot A: Ground Truth Seam
    sc1 = axes[0].scatter(points[:, 0], points[:, 1], c=s_gt, cmap='viridis', s=20)
    axes[0].set_title("Ground Truth Seam ($s_{gt}$)")
    axes[0].axis('equal'); axes[0].axis('off')
    fig.colorbar(sc1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot B: Recovered Seam
    sc2 = axes[1].scatter(points[:, 0], points[:, 1], c=s_opt, cmap='viridis', s=20)
    axes[1].set_title("Recovered Seam ($s^*$) from Noisy Metric")
    axes[1].axis('equal'); axes[1].axis('off')
    fig.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot C: Correlation Plot
    axes[2].scatter(s_gt, s_opt, alpha=0.5, s=10, c='black')
    lo = float(min(np.min(s_gt), np.min(s_opt)))
    hi = float(max(np.max(s_gt), np.max(s_opt)))
    axes[2].plot([lo, hi], [lo, hi], 'r--', lw=2)
    axes[2].plot([lo, hi], [a_fit * lo + b_fit, a_fit * hi + b_fit], color='blue', lw=2)
    axes[2].set_title("Seam Recovery Correlation")
    axes[2].set_xlabel("Ground Truth Seam Value")
    axes[2].set_ylabel("Recovered Seam Value")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    print("\nRendering visualization...")
    plt.show()

if __name__ == "__main__":
    main()