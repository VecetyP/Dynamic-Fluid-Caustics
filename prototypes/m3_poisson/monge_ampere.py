"""
M3+ · Nonlinear Monge-Ampere inverse-caustic prototype.

The paraxial solver (`poisson_caustic.inverse_caustic`) solves the LINEARISED
optics: one Poisson solve, accurate only when the target contrast is small. The
exact relation is nonlinear (mass conservation under the transport T = x + grad u):

    det(I + D^2 u) = Ibar / I                              (Monge-Ampere)

Expanding the determinant, det(I + D^2 u) = 1 + laplacian(u) + det(D^2 u), so

    laplacian(u) = (Ibar / I) - 1 - det(D^2 u)

The right side depends on u through the Hessian term, so we solve it as a damped
fixed-point (Picard) iteration: start from the paraxial solution (det(D^2 u) = 0)
and repeatedly re-solve the Poisson problem with the Hessian term folded into the
right-hand side, each solve reusing the same DCT Neumann solver. For low contrast
the Hessian term is negligible and this reduces to the paraxial answer in one
step; for high contrast it converges to the true nonlinear surface.

`forward_nonlinear` (in poisson_caustic) is the exact forward operator, so it is
the oracle we validate against: push the recovered u through it and compare the
reconstructed floor irradiance to the target.
"""

from __future__ import annotations

import numpy as np

from poisson_caustic import solve_poisson_neumann_dct, forward_nonlinear


def _hessian(u: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Second derivatives via edge-padded finite differences (same stencils the
    forward oracle uses, so the inverse and the check are consistent)."""
    up = np.pad(u, 1, mode="edge")
    uxx = (up[1:-1, 2:] - 2 * up[1:-1, 1:-1] + up[1:-1, :-2]) / (dx * dx)
    uyy = (up[2:, 1:-1] - 2 * up[1:-1, 1:-1] + up[:-2, 1:-1]) / (dx * dx)
    uxy = (up[2:, 2:] - up[2:, :-2] - up[:-2, 2:] + up[:-2, :-2]) / (4 * dx * dx)
    return uxx, uyy, uxy


def inverse_caustic_ma(
    target_I: np.ndarray,
    dx: float,
    d: float,
    n_rel: float,
    iters: int = 40,
    damping: float = 0.6,
    tol: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Nonlinear inverse: return (h_t, u, info). `info` has convergence history."""
    if np.any(target_I <= 0):
        raise ValueError("Target irradiance must be strictly positive.")
    i_bar = float(target_I.mean())
    g = i_bar / target_I  # target determinant field

    # Paraxial initial guess (one Poisson solve).
    u = solve_poisson_neumann_dct(1.0 - target_I / i_bar, dx)

    deltas = []
    for _ in range(iters):
        uxx, uyy, uxy = _hessian(u, dx)
        det_hess = uxx * uyy - uxy * uxy
        rhs = g - 1.0 - det_hess  # DCT solver drops the DC mode (Neumann mean)
        u_new = solve_poisson_neumann_dct(rhs, dx)
        step = damping * (u_new - u)
        u = u + step
        d_norm = float(np.linalg.norm(step) / (np.linalg.norm(u) + 1e-30))
        deltas.append(d_norm)
        if d_norm < tol:
            break

    h_t = -u / (d * (n_rel - 1.0))
    h_t -= h_t.mean()
    return h_t, u, {"iters": len(deltas), "last_delta": deltas[-1] if deltas else 0.0}


def _metrics(recon: np.ndarray, target_norm: np.ndarray) -> tuple[float, float]:
    """Relative L2 error and correlation of a mean-1 reconstruction vs target."""
    err = float(np.linalg.norm(recon - target_norm) / np.linalg.norm(target_norm))
    corr = float(np.corrcoef(recon.ravel(), target_norm.ravel())[0, 1])
    return err, corr


if __name__ == "__main__":
    from poisson_caustic import inverse_caustic

    n, dx, d, n_rel = 64, 0.005, 0.15, 1.333
    yy, xx = np.mgrid[0:n, 0:n] / n

    def make_target(contrast: float, kind: str = "bump") -> np.ndarray:
        """A positive target with a controllable peak/mean contrast."""
        if kind == "bump":
            base = np.exp(-(((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / (2 * 0.14 ** 2)))
        elif kind == "ring":
            r = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
            base = np.exp(-((r - 0.28) ** 2) / (2 * 0.05 ** 2))
        else:  # two blobs — a genuinely non-trivial shape
            b1 = np.exp(-(((xx - 0.35) ** 2 + (yy - 0.4) ** 2) / (2 * 0.09 ** 2)))
            b2 = np.exp(-(((xx - 0.66) ** 2 + (yy - 0.62) ** 2) / (2 * 0.11 ** 2)))
            base = np.maximum(b1, b2)
        # Map to [1, 1+contrast] so mean/peak contrast scales with `contrast`.
        return 1.0 + contrast * base

    print(f"grid {n}x{n}, dx={dx}")
    print(f"{'target':>8} {'contrast':>8} | {'paraxial err':>12} {'corr':>7} | "
          f"{'MA err':>10} {'corr':>7} {'iters':>6}")
    for kind in ("bump", "ring", "two"):
        for contrast in (0.3, 1.0, 3.0):
            I = make_target(contrast, kind)
            tnorm = I / I.mean()

            _, u_lin = inverse_caustic(I, dx, d, n_rel)
            rec_lin = forward_nonlinear(u_lin, dx)
            e_lin, c_lin = _metrics(rec_lin, tnorm)

            _, u_ma, info = inverse_caustic_ma(I, dx, d, n_rel)
            rec_ma = forward_nonlinear(u_ma, dx)
            e_ma, c_ma = _metrics(rec_ma, tnorm)

            print(f"{kind:>8} {contrast:>8.1f} | {e_lin:>12.3%} {c_lin:>7.4f} | "
                  f"{e_ma:>10.3%} {c_ma:>7.4f} {info['iters']:>6}")
