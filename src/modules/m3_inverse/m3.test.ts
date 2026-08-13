import { describe, it, expect } from "vitest";
import { InverseCausticSolver, forwardNonlinear } from "./index.ts";

// Mirrors prototypes/m3_poisson/validate.py so the TS port matches the NumPy
// oracle: (1) the solve satisfies the discrete Poisson equation, and (2) the
// analytic forward reconstruction converges quadratically as the target softens.

function syntheticTarget(n: number, contrast: number): Float64Array {
  const img = new Float64Array(n * n);
  const blob = (x: number, y: number, cx: number, cy: number, s: number) =>
    Math.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s)));
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const x = i / n;
      const y = j / n;
      img[j * n + i] =
        1 +
        contrast * blob(x, y, 0.35, 0.4, 0.1) +
        contrast * blob(x, y, 0.68, 0.62, 0.07) +
        0.5 * contrast * blob(x, y, 0.55, 0.3, 0.05);
    }
  }
  return img;
}

function laplacian5(u: Float64Array, n: number, dx: number): Float64Array {
  const at = (x: number, y: number) => {
    const cx = Math.max(0, Math.min(n - 1, x));
    const cy = Math.max(0, Math.min(n - 1, y));
    return u[cy * n + cx];
  };
  const out = new Float64Array(n * n);
  const invDx2 = 1 / (dx * dx);
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++)
      out[y * n + x] =
        (at(x + 1, y) + at(x - 1, y) + at(x, y + 1) + at(x, y - 1) - 4 * at(x, y)) * invDx2;
  return out;
}

function relL2(a: Float64Array, b: Float64Array): number {
  let num = 0;
  let den = 0;
  for (let k = 0; k < a.length; k++) {
    const d = a[k] - b[k];
    num += d * d;
    den += b[k] * b[k];
  }
  return Math.sqrt(num / den);
}

const N = 64; // smaller than the 128 default to keep the O(N³) test fast
const DX = 0.01;
const D = 0.15;
const NREL = 1.333;

describe("M3 inverse-caustic (DCT Poisson port)", () => {
  it("solution satisfies the discrete Poisson equation (residual ~ machine eps)", () => {
    const solver = new InverseCausticSolver(N);
    const I = syntheticTarget(N, 0.15);
    const { u } = solver.solve(I, DX, D, NREL);

    let iBar = 0;
    for (const v of I) iBar += v;
    iBar /= I.length;
    const rhs = new Float64Array(N * N);
    for (let k = 0; k < rhs.length; k++) rhs[k] = 1 - I[k] / iBar;

    const residual = relL2(laplacian5(u, N, DX), rhs);
    expect(residual).toBeLessThan(1e-9);
  });

  it("heightmap has zero mean and matches the TargetHeightmap contract", () => {
    const solver = new InverseCausticSolver(N);
    const { target } = solver.solve(syntheticTarget(N, 0.1), DX, D, NREL);
    expect(target.np).toBe(N);
    expect(target.hT.length).toBe(N * N);
    expect(target.d).toBe(D);
    expect(target.nRel).toBe(NREL);
    let mean = 0;
    for (const h of target.hT) mean += h;
    mean /= target.hT.length;
    expect(Math.abs(mean)).toBeLessThan(1e-6);
  });

  it("analytic forward reconstruction converges quadratically (order ~2)", () => {
    const solver = new InverseCausticSolver(N);
    const contrasts = [0.2, 0.1, 0.05, 0.025];
    const errs = contrasts.map((c) => {
      const I = syntheticTarget(N, c);
      const { u } = solver.solve(I, DX, D, NREL);
      let iBar = 0;
      for (const v of I) iBar += v;
      iBar /= I.length;
      const targetNorm = new Float64Array(I.length);
      for (let k = 0; k < I.length; k++) targetNorm[k] = I[k] / iBar;
      return relL2(forwardNonlinear(u, N, DX), targetNorm);
    });

    // Each halving of contrast should cut error ~4× (order 2).
    const orders: number[] = [];
    for (let i = 1; i < errs.length; i++) orders.push(Math.log2(errs[i - 1] / errs[i]));
    const meanOrder = orders.reduce((a, b) => a + b, 0) / orders.length;
    expect(meanOrder).toBeGreaterThan(1.8);
    expect(meanOrder).toBeLessThan(2.2);

    // Absolute accuracy sanity at the gentlest target.
    expect(errs[errs.length - 1]).toBeLessThan(1e-3);
  });
});
