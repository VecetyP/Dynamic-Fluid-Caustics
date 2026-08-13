/**
 * M3 · Inverse Caustic Solver.
 *
 * Given a target irradiance map I (from M2), compute the refractive surface
 * heightmap h_t that redistributes uniform incoming light into I. Runtime path
 * is the paraxial Poisson solve (spec §4.1.2); see the validated reference and
 * derivation in `prototypes/m3_poisson/`.
 *
 *   ∇²u = 1 − I/Ī        (Neumann BC, eq 4.2)
 *   h_t = −u / [d(n_rel−1)]   (eq 4.3)
 */

import type { TargetHeightmap } from "../../contracts/index.ts";
import { makePoissonSolver, type PoissonSolver } from "./poisson.ts";

export interface InverseCausticResult {
  target: TargetHeightmap;
  /** Transport potential u (row-major, n*n) — retained for tests/diagnostics. */
  u: Float64Array;
}

export class InverseCausticSolver {
  readonly n: number;
  private readonly solver: PoissonSolver;

  constructor(n: number) {
    this.n = n;
    this.solver = makePoissonSolver(n);
  }

  /**
   * @param I    target irradiance, row-major length n*n, strictly positive.
   * @param dx   solver cell size (m).
   * @param d    focal distance surface→floor (m).
   * @param nRel relative refractive index n2/n1.
   */
  solve(I: Float64Array, dx: number, d: number, nRel: number): InverseCausticResult {
    const n = this.n;
    if (I.length !== n * n) throw new Error(`I must be length ${n * n}`);

    let sum = 0;
    for (let k = 0; k < I.length; k++) {
      if (I[k] <= 0) throw new Error("Target irradiance must be strictly positive.");
      sum += I[k];
    }
    const iBar = sum / I.length;

    // RHS = 1 − I/Ī ; mean is exactly 0 (Neumann compatibility) by construction.
    const rhs = new Float64Array(n * n);
    for (let k = 0; k < rhs.length; k++) rhs[k] = 1 - I[k] / iBar;

    const u = this.solver.solve(rhs, dx);

    // h_t = −u / [d(n_rel−1)], then centre to zero mean (free constant).
    const scale = -1 / (d * (nRel - 1));
    const hT = new Float32Array(n * n);
    let hSum = 0;
    for (let k = 0; k < u.length; k++) {
      const h = u[k] * scale;
      hT[k] = h;
      hSum += h;
    }
    const hMean = hSum / hT.length;
    for (let k = 0; k < hT.length; k++) hT[k] -= hMean;

    return { target: { np: n, hT, d, nRel }, u };
  }
}

/**
 * Analytic forward reconstruction from the transport Jacobian — the numerical
 * oracle used to validate the solve (spec §4.1.1 mass conservation):
 *
 *   I_recon = Ī / det(I + D²u)
 *
 * Evaluated from the finite-difference Hessian of u WITHOUT re-linearising, so
 * the discrepancy vs the target is exactly the paraxial error (→0 as O(contrast²)).
 * Returns the reconstruction normalised to mean 1 (compare against I/Ī). This is
 * NOT the render path — M6 does the real ray splat; this is for tests.
 */
export function forwardNonlinear(u: Float64Array, n: number, dx: number): Float64Array {
  const at = (x: number, y: number): number => {
    const cx = x < 0 ? 0 : x > n - 1 ? n - 1 : x;
    const cy = y < 0 ? 0 : y > n - 1 ? n - 1 : y;
    return u[cy * n + cx];
  };
  const invDx2 = 1 / (dx * dx);
  const recon = new Float64Array(n * n);
  let sum = 0;
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const c = at(x, y);
      const uxx = (at(x + 1, y) - 2 * c + at(x - 1, y)) * invDx2;
      const uyy = (at(x, y + 1) - 2 * c + at(x, y - 1)) * invDx2;
      const uxy =
        (at(x + 1, y + 1) - at(x + 1, y - 1) - at(x - 1, y + 1) + at(x - 1, y - 1)) *
        (0.25 * invDx2);
      const det = (1 + uxx) * (1 + uyy) - uxy * uxy;
      const r = 1 / det;
      recon[y * n + x] = r;
      sum += r;
    }
  }
  const mean = sum / recon.length;
  for (let k = 0; k < recon.length; k++) recon[k] /= mean;
  return recon;
}
