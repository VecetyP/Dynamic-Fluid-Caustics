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
    """Return `count` grid indices placed SYMMETRICALLY around the tank perimeter.

    `count` must be a multiple of 4: it is split evenly across the four sides, and
    each side gets the same interior positions (no pistons on the corners, so no
    two land on nearly the same spot). The four sides are populated in a
    rotationally-consistent order, so the layout is symmetric under 90° rotation —
    which is what makes the wavemakers look evenly spaced on every wall.
    """
    if count % 4 != 0:
        raise ValueError(f"piston count must be a multiple of 4 for symmetry, got {count}")
    m = count // 4  # pistons per side
    # Interior positions along a side of length n (coords 0..n-1), evenly spaced
    # and symmetric about the side midpoint (endpoints excluded ⇒ no corners).
    ts = [int(round((j + 1) / (m + 1) * (n - 1))) for j in range(m)]
    cells = []
    for t in ts:
        cells.append((0, t))            # top    (row 0)
    for t in ts:
        cells.append((t, n - 1))        # right  (col n-1)
    for t in ts:
        cells.append((n - 1, n - 1 - t))  # bottom (row n-1), mirrored for rotation
    for t in ts:
        cells.append((n - 1 - t, 0))    # left   (col 0),   mirrored for rotation
    cells = np.array(cells)
    return cells[:, 0] * n + cells[:, 1]


def forward_playback(piston_cells, a, n, dx, c2dt2, damp) -> np.ndarray:
    """Replay an actuation forward and return the surface at focal time T.

    ACTUATION MODEL (the canonical one the whole pipeline agrees on): each step,
    advance the free leapfrog, THEN additively inject each piston's amplitude for
    that step into the freshly-computed h_next at its cell. `a` has shape (P, T).

    This is the exact operator the basis is built from (see build_basis_matrix),
    so M @ a.ravel() == forward_playback(a) to machine precision, and the M5 GPU/
    CPU sim replays a PistonSchedule with this identical rule.
    """
    P, T = a.shape
    h_prev = np.zeros((n, n))
    h_curr = np.zeros((n, n))
    for step in range(T):
        h_next = _step(h_curr, h_prev, dx, c2dt2, damp)
        for k, cell in enumerate(piston_cells):
            h_next.flat[cell] += a[k, step]
        h_prev, h_curr = h_curr, h_next
    return h_curr.ravel()


def _impulse_snapshots(n, piston_idx, dx, c2dt2, damp, T) -> np.ndarray:
    """Field snapshots R[m] (m=1..T) after injecting a unit into h_next from one
    piston at step 0 only, then free evolution."""
    h_prev = np.zeros((n, n))
    h_curr = np.zeros((n, n))
    R = np.empty((T, n * n))
    for step in range(T):
        h_next = _step(h_curr, h_prev, dx, c2dt2, damp)
        if step == 0:
            h_next.flat[piston_idx] += 1.0
        R[step] = h_next.ravel()          # R[step] = field at time step+1
        h_prev, h_curr = h_curr, h_next
    return R


def build_basis_matrix(n, dx, c2dt2, damp, piston_cells, T) -> np.ndarray:
    """Assemble M ∈ R^{N × P·T}, column (k,τ) = surface at focal time T from a
    unit injection by piston k at step τ. By time-shift invariance of the free
    evolution this is R_k[T−1−τ], so one sim per piston suffices (verified to
    match brute-force column-by-column playback to machine precision)."""
    P = len(piston_cells)
    M = np.empty((n * n, P * T))
    for kp, pidx in enumerate(piston_cells):
        R = _impulse_snapshots(n, pidx, dx, c2dt2, damp, T)
        for tau in range(T):
            M[:, kp * T + tau] = R[T - 1 - tau]
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
