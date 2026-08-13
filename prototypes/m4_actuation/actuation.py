"""
M4 · Wave Actuation Mapper — NumPy prototype (spec §4.3).

The forward wave sim is linear, so the surface at focal time T is a superposition
of basis waves b_k(x,t): the response to a unit impulse from piston k. An
actuation where piston k emits amplitude a_k(τ) at time τ gives (eq 4.9):

    h(x, T) = Σ_k Σ_τ a_k(τ) · b_k(x, T−τ)

Flattening surface samples into rows and (piston, emission-time) pairs into
columns turns this into a single matrix equation (eq 4.10):

    M a = h_t         M ∈ R^{N × P·T}

Generally rectangular and ill-conditioned, so we solve it as a Tikhonov-
regularised least squares (eq 4.11) — reconstruct the target while penalising
physically implausible high-energy actuations:

    a* = argmin ‖M a − h_t‖² + λ‖a‖²      ⇒   (MᵀM + λI) a* = Mᵀ h_t   (eq 4.12)

M depends only on tank geometry (not the sketch), so the regularised pseudoinverse
is pre-computed once. The connection to the outline's "time-reversal" intuition is
exact: Mᵀ h_t is the adjoint / matched filter — the first iteration of the solve —
and (MᵀM + λI)⁻¹ is the correction that turns it into the true least-squares inverse.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Forward wave stepping (leapfrog, eq 4.7) — reflective walls
# --------------------------------------------------------------------------- #
def _laplacian(H: np.ndarray, dx: float) -> np.ndarray:
    up = np.pad(H, 1, mode="edge")  # edge-replicate ⇒ ∂h/∂n = 0 (reflective)
    return (up[2:, 1:-1] + up[:-2, 1:-1] + up[1:-1, 2:] + up[1:-1, :-2] - 4 * up[1:-1, 1:-1]) / (dx * dx)


def _step(h_curr, h_prev, dx, c2dt2, damp):
    inv = 1.0 / (1.0 + damp)
    return inv * (2.0 * h_curr + (damp - 1.0) * h_prev + c2dt2 * _laplacian(h_curr, dx))


def perimeter_pistons(n: int, count: int) -> np.ndarray:
    """Return `count` grid indices spaced evenly around the tank perimeter ring."""
    ring = []
    for i in range(n):
        ring.append((0, i))          # top
    for j in range(1, n):
        ring.append((j, n - 1))      # right
    for i in range(n - 2, -1, -1):
        ring.append((n - 1, i))      # bottom
    for j in range(n - 2, 0, -1):
        ring.append((j, 0))          # left
    ring = np.array(ring)
    sel = np.linspace(0, len(ring) - 1, count).round().astype(int)
    sel = np.unique(sel)
    cells = ring[sel]
    return cells[:, 0] * n + cells[:, 1]


def impulse_response(n, piston_idx, dx, c2dt2, damp, T):
    """Snapshots of piston k's impulse response at ages 1..T (a unit displacement
    delta at the piston cell, zero initial velocity, then free evolution)."""
    grid = np.zeros((n, n))
    grid.flat[piston_idx] = 1.0
    h_prev = grid.copy()
    h_curr = grid.copy()
    snaps = np.empty((T, n * n))
    for s in range(T):
        h_next = _step(h_curr, h_prev, dx, c2dt2, damp)
        snaps[s] = h_next.ravel()
        h_prev, h_curr = h_curr, h_next
    return snaps  # snaps[a-1] = field at age a


def build_basis_matrix(n, dx, c2dt2, damp, piston_cells, T) -> np.ndarray:
    """Assemble M ∈ R^{N × P·T}. Column (k,τ) = b_k(·, T−τ), i.e. piston k's
    impulse response at age (T−τ). One forward sim per piston."""
    P = len(piston_cells)
    M = np.empty((n * n, P * T))
    for kp, pidx in enumerate(piston_cells):
        snaps = impulse_response(n, pidx, dx, c2dt2, damp, T)
        for tau in range(T):
            age = T - tau                 # age ∈ [1, T]
            M[:, kp * T + tau] = snaps[age - 1]
    return M


# --------------------------------------------------------------------------- #
# Solvers
# --------------------------------------------------------------------------- #
def solve_least_squares(M, h_t, lam, MtM=None, Mt_h=None):
    """Tikhonov-regularised least squares a* = (MᵀM + λI)⁻¹ Mᵀ h_t (eq 4.12).
    Pass precomputed MtM / Mt_h to reuse across a λ sweep."""
    if MtM is None:
        MtM = M.T @ M
    if Mt_h is None:
        Mt_h = M.T @ h_t
    A = MtM + lam * np.eye(MtM.shape[0])
    return np.linalg.solve(A, Mt_h)


def solve_adjoint(M, h_t, Mt_h=None):
    """Raw adjoint / time-reversal a = Mᵀ h_t, scaled to best-fit amplitude.
    This is the matched filter — the first iteration of the least-squares solve."""
    if Mt_h is None:
        Mt_h = M.T @ h_t
    a = Mt_h.copy()
    forward = M @ a
    denom = float(forward @ forward)
    alpha = float(h_t @ forward) / denom if denom > 0 else 0.0
    return alpha * a
