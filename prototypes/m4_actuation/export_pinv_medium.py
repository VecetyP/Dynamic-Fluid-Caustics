"""
Export the MEDIUM (32x32 / 24-piston) actuation pseudoinverse for the runtime.

Same math as export_pinv.py, but at a higher resolution whose M+ is ~3.9 MB —
too big to inline as JSON, so the pinv floats are written as a raw little-endian
float32 BINARY (`pinv_medium.bin`) and the geometry + golden sample go in a small
JSON sidecar (`pinv_medium.json`). The runtime loads the sidecar for geometry and
fetches the .bin for the matrix (ActuationMapper.fromBinary).

Run:  python export_pinv_medium.py
"""

from __future__ import annotations

import json
import os
import numpy as np

from actuation import perimeter_pistons, build_basis_matrix, forward_playback

GRAVITY = 9.81


def build(n=32, dx=0.01, depth=0.05, gamma=0.2, P=24, T=40, lam=1e-2):
    c = np.sqrt(GRAVITY * depth)
    dt = 0.9 * dx / (c * np.sqrt(2))
    c2dt2 = c * c * dt * dt
    damp = 0.5 * gamma * dt

    cells = perimeter_pistons(n, P)
    M = build_basis_matrix(n, dx, c2dt2, damp, cells, T)
    A = M.T @ M + lam * np.eye(M.shape[1])
    pinv = np.linalg.solve(A, M.T)  # (P*T) x N

    # Golden sample: a smooth zero-mean bump and its recovered actuation.
    yy, xx = np.mgrid[0:n, 0:n] / n
    h_t = np.exp(-(((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / (2 * 0.12 ** 2)))
    h_t = (h_t - h_t.mean()).ravel()
    a_star = pinv @ h_t

    geometry = {
        "n": int(n), "dx": float(dx), "depth": float(depth), "gamma": float(gamma),
        "pistonCount": int(len(cells)), "numSteps": int(T), "lambda": float(lam),
        "dt": float(dt), "focalTime": float(T * dt),
        "pistonCells": [int(v) for v in cells],
    }
    return geometry, pinv, h_t, a_star, cells, (c2dt2, damp)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.normpath(
        os.path.join(here, "..", "..", "src", "modules", "m4_actuation", "__fixtures__")
    )
    os.makedirs(out_dir, exist_ok=True)

    geometry, pinv, h_t, a_star, cells, (c2dt2, damp) = build()

    # Binary: row-major float32, little-endian.
    bin_path = os.path.join(out_dir, "pinv_medium.bin")
    pinv.astype("<f4").tofile(bin_path)

    # JSON sidecar: geometry + golden sample only (no pinv).
    json_path = os.path.join(out_dir, "pinv_medium.json")
    with open(json_path, "w") as f:
        json.dump({
            "geometry": geometry,
            "sample": {
                "hT": [float(v) for v in h_t],
                "aExpected": [float(v) for v in a_star],
            },
        }, f)

    # Closed-loop sanity.
    a = a_star.reshape(geometry["pistonCount"], geometry["numSteps"])
    play = forward_playback(cells, a, geometry["n"], geometry["dx"], c2dt2, damp)
    rel = np.linalg.norm(play - h_t) / np.linalg.norm(h_t)
    corr = np.corrcoef(play, h_t)[0, 1]

    print("exported MEDIUM M+ asset")
    print(f"  geometry: n={geometry['n']} pistons={geometry['pistonCount']} "
          f"T={geometry['numSteps']} lambda={geometry['lambda']}")
    print(f"  pinv: {pinv.shape[0]}x{pinv.shape[1]}  "
          f"({os.path.getsize(bin_path)/1e6:.2f} MB bin, {os.path.getsize(json_path)/1e3:.0f} KB json)")
    print(f"  closed-loop reconstructs h_t: rel_err={rel:.3e} corr={corr:.4f}")
    print(f"  -> {bin_path}")
    print(f"  -> {json_path}")


if __name__ == "__main__":
    main()
