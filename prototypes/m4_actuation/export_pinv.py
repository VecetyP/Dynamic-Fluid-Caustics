"""
Export a precomputed regularised pseudoinverse M⁺ for the TS runtime.

M⁺ = (MᵀM + λI)⁻¹ Mᵀ depends only on tank geometry (spec §4.3), so it is built
ONCE here and shipped as an asset; the app then turns any target surface into a
PistonSchedule with a single matrix-vector product a = M⁺ · h_t.

Writes a JSON asset with the pseudoinverse, geometry metadata, and a golden
sample (h_t, a* = M⁺·h_t) so the TS port can be regression-tested against NumPy.

This fixture uses a deliberately SMALL tank so the asset stays lightweight and
the test is fast; the matvec agreement it verifies is geometry-independent. A
production-sized M⁺ (128²) is large — see PROGRESS.md notes on compression/
streaming — and out of scope for the port's correctness check.

Run:  python export_pinv.py
"""

from __future__ import annotations

import json
import os
import numpy as np

from actuation import perimeter_pistons, build_basis_matrix, forward_playback

GRAVITY = 9.81


def build_export(n=16, dx=0.02, depth=0.05, gamma=0.2, P=12, T=20, lam=1e-2):
    c = np.sqrt(GRAVITY * depth)
    dt = 0.9 * dx / (c * np.sqrt(2))          # 0.9× CFL (eq 4.8)
    c2dt2 = c * c * dt * dt
    damp = 0.5 * gamma * dt

    cells = perimeter_pistons(n, P)
    piston_count = len(cells)

    M = build_basis_matrix(n, dx, c2dt2, damp, cells, T)   # N × (P·T)
    MtM = M.T @ M
    A = MtM + lam * np.eye(MtM.shape[0])
    pinv = np.linalg.solve(A, M.T)                          # (P·T) × N  == M⁺

    # Golden sample: a smooth zero-mean bump and its recovered actuation.
    yy, xx = np.mgrid[0:n, 0:n] / n
    h_t = np.exp(-(((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / (2 * 0.12 ** 2)))
    h_t = (h_t - h_t.mean()).ravel()
    a_star = pinv @ h_t

    return {
        "geometry": {
            "n": int(n),
            "dx": float(dx),
            "depth": float(depth),      # sets wave speed c=√(gH) for the forward sim
            "gamma": float(gamma),      # damping γ
            "pistonCount": int(piston_count),
            "numSteps": int(T),
            "lambda": float(lam),
            "dt": float(dt),
            "focalTime": float(T * dt),
            "pistonCells": [int(v) for v in cells],
        },
        "pinv": {
            "rows": int(pinv.shape[0]),   # P·T
            "cols": int(pinv.shape[1]),   # N
            "data": [float(v) for v in pinv.ravel(order="C")],
        },
        "sample": {
            "hT": [float(v) for v in h_t],
            "aExpected": [float(v) for v in a_star],
        },
    }


def main() -> None:
    doc = build_export()
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.normpath(
        os.path.join(here, "..", "..", "src", "modules", "m4_actuation", "__fixtures__", "pinv_small.json")
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(doc, f)
    g = doc["geometry"]
    size_kb = os.path.getsize(out) / 1024

    # Closed-loop sanity: replay the exported schedule, check it reconstructs h_t.
    c = np.sqrt(GRAVITY * g["depth"])
    c2dt2 = c * c * g["dt"] * g["dt"]
    damp = 0.5 * g["gamma"] * g["dt"]
    a = np.array(doc["sample"]["aExpected"]).reshape(g["pistonCount"], g["numSteps"])
    h_t = np.array(doc["sample"]["hT"])
    play = forward_playback(g["pistonCells"], a, g["n"], g["dx"], c2dt2, damp)
    rel = np.linalg.norm(play - h_t) / np.linalg.norm(h_t)
    corr = np.corrcoef(play, h_t)[0, 1]

    print("exported M⁺ asset")
    print(f"  geometry: n={g['n']} pistons={g['pistonCount']} T={g['numSteps']} "
          f"λ={g['lambda']}")
    print(f"  pinv: {doc['pinv']['rows']}×{doc['pinv']['cols']}  "
          f"({size_kb:.0f} KB JSON)")
    print(f"  closed-loop forward-sim reconstructs hₜ: rel_err={rel:.3e} corr={corr:.4f}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
