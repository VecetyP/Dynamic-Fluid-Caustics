import { describe, it, expect } from "vitest";
import { preprocessDensity, gaussianBlur, DEFAULT_DENSITY_OPTIONS } from "./index.ts";

const N = 16;

function spike(n: number): Float32Array {
  const a = new Float32Array(n * n);
  a[(n / 2) * n + n / 2] = 1; // single bright cell
  return a;
}

describe("M2 density preprocessor", () => {
  it("produces a strictly positive target (spec invariant I >= EPS)", () => {
    const I = preprocessDensity(spike(N), N).I;
    for (const v of I) expect(v).toBeGreaterThan(0);
  });

  it("iBar equals the mean of I (uniform source level)", () => {
    const d = preprocessDensity(spike(N), N);
    let mean = 0;
    for (const v of d.I) mean += v;
    mean /= d.I.length;
    expect(Math.abs(d.iBar - mean)).toBeLessThan(1e-6);
  });

  it("blur spreads energy while conserving the total", () => {
    const src = spike(N);
    const blurred = gaussianBlur(src, N, DEFAULT_DENSITY_OPTIONS.blurSigma);
    const sum = (a: Float32Array) => a.reduce((s, v) => s + v, 0);
    // Gaussian is normalised ⇒ total intensity preserved.
    expect(Math.abs(sum(blurred) - sum(src))).toBeLessThan(1e-4);
    // The single spike is spread to neighbours (center reduced, neighbour raised).
    expect(blurred[(N / 2) * N + N / 2]).toBeLessThan(src[(N / 2) * N + N / 2]);
    expect(blurred[(N / 2) * N + N / 2 + 1]).toBeGreaterThan(0);
  });

  it("rejects a mismatched input length", () => {
    expect(() => preprocessDensity(new Float32Array(10), N)).toThrow(/length/);
  });
});
