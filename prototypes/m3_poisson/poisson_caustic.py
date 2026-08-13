"""
M3 · Inverse-caustic solver — NumPy/SciPy prototype (spec §4.1.2).

This is the Phase-2 validation prototype. It solves the *real-time* (paraxial)
inverse-caustic step and forward-checks it, before any GPU/WGSL port. The exact
Monge-Ampere / optimal-transport path (§4.1.1) is deliberately NOT implemented
here — the spec ships Poisson as the interactive default.

Math
----
Uniform incoming light illuminates a refractive surface h(x,y); we want the
surface that redistributes that light so the floor shows target irradiance I.

Paraxial linearisation (spec eq 4.2), with Ibar = mean(I):

    ∇²u = 1 − I/Ibar         (homogeneous Neumann BC: no transport across walls)

The transport map is T(x) = x + ∇u(x); the surface follows directly (eq 4.3):

    h_t = −u / [ d (n_rel − 1) ]   (+ const)

Neumann compatibility: ∫(1 − I/Ibar) = 0 holds automatically because Ibar is the
mean of I — this is exactly why the source level is defined that way.

Solver
------
Constant-coefficient Laplacian on a regular grid with Neumann BCs diagonalises
under the DCT-II. We transform the RHS, divide by the stencil eigenvalues, and
invert — an O(N log N) solve, the sub-millisecond runtime path (spec §4.1.2).
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn


# --------------------------------------------------------------------------- #
# Core solver
# --------------------------------------------------------------------------- #
def solve_poisson_neumann_dct(rhs: np.ndarray, dx: float) -> np.ndarray:
    """Solve ∇²u = rhs with homogeneous Neumann BCs via the DCT.

    Uses the eigenvalues of the 5-point Laplacian stencil under DCT-II so the
    discrete solution matches the same stencil M5 integrates. The nullspace
    (additive constant) is fixed by setting the DC mode to zero.
    """
    n_y, n_x = rhs.shape

    # Forward DCT-II (orthonormal) of the right-hand side.
    rhs_hat = dctn(rhs, type=2, norm="ortho")

    # Eigenvalues of the 5-point Laplacian for DCT-II modes:
    #   λ_i = 2(cos(π i / N) − 1) / dx²   (per axis; here Δx == Δy)
    i = np.arange(n_x)
    j = np.arange(n_y)
    lam_x = (2.0 * (np.cos(np.pi * i / n_x) - 1.0)) / (dx * dx)
    lam_y = (2.0 * (np.cos(np.pi * j / n_y) - 1.0)) / (dx * dx)
    denom = lam_y[:, None] + lam_x[None, :]

    # λ(0,0) == 0 is the undetermined constant; avoid div-by-zero, set DC = 0.
    denom[0, 0] = 1.0
    u_hat = rhs_hat / denom
    u_hat[0, 0] = 0.0

    u = idctn(u_hat, type=2, norm="ortho")
    return u


def inverse_caustic(
    target_I: np.ndarray, dx: float, d: float, n_rel: float
) -> tuple[np.ndarray, np.ndarray]:
    """Given target irradiance I, return (target heightmap h_t, potential u).

    h_t is centred to zero mean (the additive constant is physically free).
    """
    if np.any(target_I <= 0):
        raise ValueError("Target irradiance must be strictly positive (spec invariant).")

    i_bar = float(target_I.mean())
    rhs = 1.0 - target_I / i_bar  # eq 4.2 RHS; mean(rhs) == 0 by construction
    u = solve_poisson_neumann_dct(rhs, dx)

    h_t = -u / (d * (n_rel - 1.0))  # eq 4.3
    h_t -= h_t.mean()
    return h_t, u


# --------------------------------------------------------------------------- #
# Forward checks
# --------------------------------------------------------------------------- #
def forward_nonlinear(u: np.ndarray, dx: float) -> np.ndarray:
    """Reconstruct floor irradiance analytically from the transport Jacobian.

    The exact redistribution is I = Ibar / det(I + D²u) (mass conservation under
    T = x + ∇u, spec §4.1.1). Evaluating it from the finite-difference Hessian of
    the *recovered* u — WITHOUT re-linearising — is the clean numerical oracle:
    the discrepancy vs the target is precisely the paraxial linearisation error,
    and it must vanish as O(contrast²). No Monte-Carlo noise, unlike the ray splat.

    Returns the reconstruction normalised to mean 1 (compare against I / Ibar).
    """
    up = np.pad(u, 1, mode="edge")
    uxx = (up[1:-1, 2:] - 2 * up[1:-1, 1:-1] + up[1:-1, :-2]) / (dx * dx)
    uyy = (up[2:, 1:-1] - 2 * up[1:-1, 1:-1] + up[:-2, 1:-1]) / (dx * dx)
    uxy = (up[2:, 2:] - up[2:, :-2] - up[:-2, 2:] + up[:-2, :-2]) / (4 * dx * dx)
    det = (1.0 + uxx) * (1.0 + uyy) - uxy * uxy
    recon = 1.0 / det
    return recon / recon.mean()


def forward_paraxial(u: np.ndarray, dx: float, supersample: int = 4) -> np.ndarray:
    """Reconstruct floor irradiance by pushing uniform light through T = x + ∇u.

    Emit `supersample²` sub-samples per cell, displace each by the local ∇u, and
    histogram the landing positions back onto the grid. In the linear regime this
    must reproduce the target I that produced u — that is the validation.
    """
    n_y, n_x = u.shape

    # ∇u by central differences (matches the solver's stencil family).
    gy, gx = np.gradient(u, dx)

    # Uniform grid of emitters, supersampled.
    s = supersample
    lin = (np.arange(n_x * s) + 0.5) / s - 0.5  # cell-centred sub-sample coords
    liny = (np.arange(n_y * s) + 0.5) / s - 0.5
    sx, sy = np.meshgrid(lin, liny)

    # Bilinear sample of the displacement field at sub-sample positions.
    dispx = _bilinear(gx, sx, sy)
    dispy = _bilinear(gy, sx, sy)

    # Transport map T(x) = x + ∇u  (u carries dx² units → convert to cells).
    tx = sx + dispx / dx
    ty = sy + dispy / dx

    # Histogram landing points onto the base grid; normalise to mean 1 for
    # comparison with I/Ibar.
    hist, _, _ = np.histogram2d(
        ty.ravel(), tx.ravel(),
        bins=[n_y, n_x],
        range=[[-0.5, n_y - 0.5], [-0.5, n_x - 0.5]],
    )
    hist /= hist.mean()
    return hist


def _bilinear(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample of `field` at fractional (x, y) cell coords, clamped."""
    n_y, n_x = field.shape
    x0 = np.clip(np.floor(x).astype(int), 0, n_x - 1)
    y0 = np.clip(np.floor(y).astype(int), 0, n_y - 1)
    x1 = np.clip(x0 + 1, 0, n_x - 1)
    y1 = np.clip(y0 + 1, 0, n_y - 1)
    fx = np.clip(x - x0, 0.0, 1.0)
    fy = np.clip(y - y0, 0.0, 1.0)
    return (
        field[y0, x0] * (1 - fx) * (1 - fy)
        + field[y0, x1] * fx * (1 - fy)
        + field[y1, x0] * (1 - fx) * fy
        + field[y1, x1] * fx * fy
    )
