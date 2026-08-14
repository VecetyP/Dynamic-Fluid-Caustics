"""
R2 · High-resolution actuation sweep.

Question: at a finer grid with more pistons, how many time steps T does the
schedule need before the wall pistons can actually build a target surface at the
tank centre? (With a finer grid on the same physical tank, waves take more steps
to travel in from the walls.) Also: how big does the precomputed pseudoinverse
get, to decide how to ship it.

For each T we build the wave-basis M, solve the regularised least squares for a
realistic Monge-Ampere target surface, replay it forward, and report the
reconstruction quality plus the pseudoinverse size.
"""

from __future__ import annotations

import gc
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "m3_poisson"))
from monge_ampere import inverse_caustic_ma  # noqa: E402
from actuation import (  # noqa: E402
    perimeter_pistons,
    build_basis_matrix,
    solve_least_squares,
    forward_playback,
)

GRAVITY = 9.81


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main():
    n, dx, depth, gamma = 64, 0.005, 0.05, 0.2
    P, lam = 48, 1e-2
    d, n_rel = 0.15, 1.333

    c = np.sqrt(GRAVITY * depth)
    dt = 0.9 * dx / (c * np.sqrt(2))
    c2dt2 = c * c * dt * dt
    damp = 0.5 * gamma * dt
    cross_steps = (n * dx) / c / dt  # steps for a wave to cross the tank

    # A realistic target surface: Monge-Ampere solve of a two-blob, medium-contrast
    # image, scaled to a fixed peak so the numbers are comparable across T.
    yy, xx = np.mgrid[0:n, 0:n] / n
    b1 = np.exp(-(((xx - 0.35) ** 2 + (yy - 0.42) ** 2) / (2 * 0.10 ** 2)))
    b2 = np.exp(-(((xx - 0.65) ** 2 + (yy - 0.60) ** 2) / (2 * 0.12 ** 2)))
    I = 1.0 + 1.0 * np.maximum(b1, b2)
    h_t, _, _ = inverse_caustic_ma(I, dx, d, n_rel)
    h_t = h_t / np.max(np.abs(h_t))  # unit peak
    h_vec = h_t.ravel()

    print(f"grid {n}x{n}, dx={dx} m, tank={n*dx:.2f} m, c={c:.3f} m/s, dt={dt:.5f} s")
    print(f"wave crosses the tank in ~{cross_steps:.0f} steps")
    print(f"pistons P={P} ({P//4}/side), lambda={lam}\n")
    print(f"{'T':>5} {'focal(s)':>9} {'recon err':>10} {'corr':>7} "
          f"{'M+ floats':>12} {'M+ f32':>9} {'build':>7}")

    import time
    for T in (60, 100, 140, 180):
        t0 = time.time()
        cells = perimeter_pistons(n, P)
        M = build_basis_matrix(n, dx, c2dt2, damp, cells, T)  # N x (P*T)
        a = solve_least_squares(M, h_vec, lam)
        recon = forward_playback(cells, a.reshape(P, T), n, dx, c2dt2, damp)
        e = rel_l2(recon, h_vec)
        c_ = corr(recon, h_vec)
        rows, cols = P * T, n * n
        floats = rows * cols
        mb = floats * 4 / 1e6
        dtb = time.time() - t0
        print(f"{T:>5} {T*dt:>9.3f} {e:>10.2%} {c_:>7.4f} "
              f"{floats:>12,} {mb:>7.1f}MB {dtb:>6.1f}s")
        del M, a, recon
        gc.collect()


if __name__ == "__main__":
    main()
