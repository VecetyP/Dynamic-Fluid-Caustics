"""
Validation for the M3 Poisson inverse-caustic prototype.

Two independent checks:
  [1] Solver correctness — the recovered u satisfies the discrete Poisson eqn
      (residual ~ machine epsilon for a spectral solve).
  [2] Physical correctness — reconstruct floor irradiance from the transport
      Jacobian det(I + D²u) of the recovered surface (analytic, noise-free) and
      compare to the target. The discrepancy is the paraxial linearisation error;
      a correct first-order solver makes it vanish as O(contrast²). We verify that
      quadratic convergence across a contrast sweep.

A ray-splat forward pass (what the GPU actually does) is also rendered into the
figure for a qualitative look, but it is NOT used for pass/fail — histogram
quantisation makes it unreliable at sub-cell transport.

Run:  python validate.py
"""

from __future__ import annotations

import numpy as np

from poisson_caustic import (
    inverse_caustic,
    forward_nonlinear,
    forward_paraxial,
)


def synthetic_target(n: int, contrast: float = 0.5) -> np.ndarray:
    """Smooth, band-limited, strictly-positive target: bright blobs on a uniform
    background. Gentle slopes keep us inside the paraxial envelope (spec §6)."""
    yy, xx = np.mgrid[0:n, 0:n] / n
    def blob(cx, cy, s):
        return np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s)))
    img = np.ones((n, n))
    img += contrast * blob(0.35, 0.40, 0.10)
    img += contrast * blob(0.68, 0.62, 0.07)
    img += 0.5 * contrast * blob(0.55, 0.30, 0.05)
    return img


def laplacian_5pt(u: np.ndarray, dx: float) -> np.ndarray:
    up = np.pad(u, 1, mode="edge")
    lap = (up[2:, 1:-1] + up[:-2, 1:-1] + up[1:-1, 2:] + up[1:-1, :-2] - 4 * up[1:-1, 1:-1])
    return lap / (dx * dx)


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main() -> None:
    n = 128
    dx = 0.01       # 1 cm cells (matches sim default)
    d = 0.15        # 15 cm surface -> floor
    n_rel = 1.333   # water

    # -- Reference case for the figure + headline numbers --------------------
    contrast = 0.15
    target = synthetic_target(n, contrast)
    i_bar = target.mean()
    h_t, u = inverse_caustic(target, dx, d, n_rel)

    residual = rel_l2(laplacian_5pt(u, dx), 1.0 - target / i_bar)
    recon = forward_nonlinear(u, dx)
    target_norm = target / i_bar
    recon_err = rel_l2(recon, target_norm)
    corr = float(np.corrcoef(recon.ravel(), target_norm.ravel())[0, 1])

    gy, gx = np.gradient(h_t, dx)
    max_slope = float(np.max(np.hypot(gx, gy)))
    p2p_mm = float((h_t.max() - h_t.min()) * 1000)

    print("=== M3 Poisson inverse-caustic — validation ===")
    print(f"grid {n}x{n}  dx={dx} m  d={d} m  n_rel={n_rel}  contrast={contrast}")
    print(f"[1] Poisson residual ||∇²u−rhs||/||rhs|| = {residual:.3e}   (want <1e-9)")
    print(f"[2] Reconstruction rel-L2 = {recon_err:.3e}   corr = {corr:.5f}")
    print(f"    surface peak-to-peak = {p2p_mm:.2f} mm   max|∇hₜ| = {max_slope:.3f}")

    # -- Convergence study: paraxial error must be ~O(contrast²) --------------
    print("\nContrast sweep (expect err to drop ~4× per halving → quadratic):")
    print(f"  {'contrast':>9} {'rel-L2':>11} {'ratio':>7}")
    prev = None
    orders = []
    for c in [0.20, 0.10, 0.05, 0.025, 0.0125]:
        tgt = synthetic_target(n, c)
        _, uu = inverse_caustic(tgt, dx, d, n_rel)
        err = rel_l2(forward_nonlinear(uu, dx), tgt / tgt.mean())
        if prev is not None:
            ratio = prev / err
            orders.append(np.log2(ratio))
            print(f"  {c:9.4f} {err:11.3e} {ratio:6.2f}x")
        else:
            print(f"  {c:9.4f} {err:11.3e} {'—':>6}")
        prev = err
    mean_order = float(np.mean(orders))
    print(f"  mean observed order p ≈ {mean_order:.2f}  (2.0 == ideal quadratic)")

    solver_ok = residual < 1e-9
    conv_ok = mean_order > 1.8
    print(f"\nRESULT: {'PASS' if (solver_ok and conv_ok) else 'FAIL'}"
          f"  (solver_ok={solver_ok}, quadratic_convergence={conv_ok})")

    _save_figure(target_norm, h_t, recon, u, dx)


def _save_figure(target_norm, h_t, recon, u, dx) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(figure skipped: {e})")
        return

    splat = forward_paraxial(u, dx, supersample=6)  # qualitative only
    fig, ax = plt.subplots(1, 5, figsize=(19, 4))
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    ims = [
        (ax[0].imshow(target_norm, cmap="magma"), "Target  I / Ī"),
        (ax[1].imshow(h_t * 1000, cmap="RdBu"), "Surface hₜ (mm)"),
        (ax[2].imshow(recon, cmap="magma"), "Reconstruct (analytic)"),
        (ax[3].imshow(recon - target_norm, cmap="coolwarm"), "Error (recon − target)"),
        (ax[4].imshow(splat, cmap="magma"), "Ray-splat (GPU-style)"),
    ]
    for a, (im, title) in zip(ax, ims):
        a.set_title(title); fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig("validation.png", dpi=110)
    print("saved comparison figure -> validation.png")


if __name__ == "__main__":
    main()
