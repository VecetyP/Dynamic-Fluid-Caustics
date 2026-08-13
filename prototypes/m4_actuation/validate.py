"""
Validation for the M4 actuation prototype.

Checks:
  [1] Reconstruction — a target focal surface h_t is reproduced by the recovered
      actuation: h_recon = M a*. Report relative error and correlation.
  [2] Regularisation trade-off — sweep λ and trace the L-curve (residual vs
      actuation energy ‖a‖). Small λ fits better but needs high energy; large λ
      is gentler but blurs. Pick the knee.
  [3] Adjoint vs least-squares — confirm the spec's claim that raw time-reversal
      (Mᵀ h_t) already gives a recognisable surface, and the regularised inverse
      sharpens it.

Run:  python validate.py
"""

from __future__ import annotations

import numpy as np

from actuation import (
    perimeter_pistons,
    build_basis_matrix,
    solve_least_squares,
    solve_adjoint,
)

# Shallow-water params (match the app defaults, scaled down for a fast prototype).
GRAVITY = 9.81


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def target_surface(n: int) -> np.ndarray:
    """A smooth low-frequency focal target the perimeter pistons can plausibly
    form: a centred Gaussian bump, zero-mean."""
    yy, xx = np.mgrid[0:n, 0:n] / n
    bump = np.exp(-(((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / (2 * 0.12 ** 2)))
    bump -= bump.mean()
    return bump.ravel()


def main() -> None:
    n = 44
    dx = 0.02
    depth = 0.05
    gamma = 0.2
    P = 28
    T = 60

    c = np.sqrt(GRAVITY * depth)
    dt = 0.9 * dx / (c * np.sqrt(2))       # 0.9× CFL bound (eq 4.8)
    c2dt2 = c * c * dt * dt
    damp = 0.5 * gamma * dt

    piston_cells = perimeter_pistons(n, P)
    print("=== M4 actuation mapper — validation ===")
    print(f"grid {n}x{n}  N={n*n}  pistons={len(piston_cells)}  T={T}  "
          f"cols=P·T={len(piston_cells)*T}")
    print(f"c={c:.3f} m/s  dt={dt:.4e} s  crossing≈{(n*dx)/c/dt:.0f} steps")

    M = build_basis_matrix(n, dx, c2dt2, damp, piston_cells, T)
    MtM = M.T @ M
    h_t = target_surface(n)
    Mt_h = M.T @ h_t
    print(f"M shape {M.shape}  cond(MᵀM)≈{np.linalg.cond(MtM):.2e}")

    # [2] λ sweep / L-curve.
    print("\nλ sweep (residual vs actuation energy):")
    print(f"  {'lambda':>10} {'rel_err':>9} {'corr':>7} {'||a||':>10}")
    lams = np.logspace(-6, 1, 8)
    rows = []
    for lam in lams:
        a = solve_least_squares(M, h_t, lam, MtM=MtM, Mt_h=Mt_h)
        recon = M @ a
        rows.append((lam, rel_l2(recon, h_t), corr(recon, h_t), float(np.linalg.norm(a))))
        print(f"  {lam:10.2e} {rows[-1][1]:9.3e} {rows[-1][2]:7.4f} {rows[-1][3]:10.3e}")

    # Operating point: the LARGEST λ that still keeps reconstruction faithful
    # (≤ TOL error). Most regularisation we can afford against the ~1e19
    # conditioning without sacrificing the caustic. More decision-relevant than
    # an abstract L-curve knee, which here sits at over-smoothed amplitudes.
    TOL = 0.05
    faithful = [i for i, r in enumerate(rows) if r[1] <= TOL]
    knee = faithful[-1] if faithful else int(np.argmin([r[1] for r in rows]))
    lam_star, err_star, corr_star, en_star = rows[knee]
    print(f"\noperating λ (largest with err≤{TOL:.0%}) = {lam_star:.2e}  "
          f"rel_err={err_star:.3e}  corr={corr_star:.4f}  ||a||={en_star:.3e}")

    # [3] Adjoint (time-reversal) vs least-squares at the knee.
    a_adj = solve_adjoint(M, h_t, Mt_h=Mt_h)
    recon_adj = M @ a_adj
    a_ls = solve_least_squares(M, h_t, lam_star, MtM=MtM, Mt_h=Mt_h)
    recon_ls = M @ a_ls
    print(f"adjoint (time-reversal): rel_err={rel_l2(recon_adj, h_t):.3e}  "
          f"corr={corr(recon_adj, h_t):.4f}")
    print(f"least-squares (λ knee) : rel_err={rel_l2(recon_ls, h_t):.3e}  "
          f"corr={corr(recon_ls, h_t):.4f}")

    ok = corr_star > 0.9 and corr(recon_adj, h_t) > 0.5
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}  "
          f"(LS recovers target; adjoint already recognisable)")

    _save_figure(n, h_t, recon_ls, recon_adj, rows, knee)


def _save_figure(n, h_t, recon_ls, recon_adj, rows, knee):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(figure skipped: {e})")
        return

    H = lambda v: v.reshape(n, n)
    fig = plt.figure(figsize=(16, 4))
    ax = [fig.add_subplot(1, 4, i + 1) for i in range(4)]
    lim = np.max(np.abs(h_t))
    for a in ax[:3]:
        a.set_xticks([]); a.set_yticks([])
    ax[0].imshow(H(h_t), cmap="RdBu", vmin=-lim, vmax=lim); ax[0].set_title("Target hₜ")
    ax[1].imshow(H(recon_ls), cmap="RdBu", vmin=-lim, vmax=lim); ax[1].set_title("Least-squares  M a*")
    ax[2].imshow(H(recon_adj), cmap="RdBu", vmin=-lim, vmax=lim); ax[2].set_title("Adjoint (time-reversal)")

    errs = [r[1] for r in rows]; ens = [r[3] for r in rows]; lams = [r[0] for r in rows]
    ax[3].loglog(errs, ens, "-o", color="#2dd4bf")
    ax[3].loglog(errs[knee], ens[knee], "o", ms=12, mfc="none", mec="crimson")
    for r in rows:
        ax[3].annotate(f"{r[0]:.0e}", (r[1], r[3]), fontsize=7, alpha=0.7)
    ax[3].set_xlabel("reconstruction error"); ax[3].set_ylabel("actuation energy ‖a‖")
    ax[3].set_title("L-curve (circle = operating λ)")
    fig.tight_layout()
    fig.savefig("validation.png", dpi=110)
    print("saved comparison figure -> validation.png")


if __name__ == "__main__":
    main()
