import { describe, it, expect } from "vitest";
import { ActuationMapper, type PinvAsset } from "./index.ts";
import assetJson from "./__fixtures__/pinv_small.json";

// The fixture is produced by prototypes/m4_actuation/export_pinv.py, which ships
// the precomputed M⁺ plus a golden sample (h_t, a* = M⁺·h_t computed in NumPy).
// This test verifies the TS runtime matvec reproduces NumPy's a* — i.e. the port
// is faithful — and that it emits a well-formed PistonSchedule.

const asset = assetJson as PinvAsset;

function relL2(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let num = 0;
  let den = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    num += d * d;
    den += b[i] * b[i];
  }
  return Math.sqrt(num / den);
}

describe("M4 actuation mapper (TS port of M⁺ matvec)", () => {
  it("reproduces the NumPy a* for the golden sample", () => {
    const mapper = new ActuationMapper(asset);
    const schedule = mapper.solve(asset.sample!.hT);

    expect(schedule.a.length).toBe(asset.sample!.aExpected.length);
    const err = relL2(schedule.a, asset.sample!.aExpected);
    expect(err).toBeLessThan(1e-6); // matvec is exact bar float rounding

    // Spot-check a few entries in absolute terms too.
    let maxAbs = 0;
    for (let i = 0; i < schedule.a.length; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(schedule.a[i] - asset.sample!.aExpected[i]));
    }
    expect(maxAbs).toBeLessThan(1e-4);
  });

  it("emits a well-formed PistonSchedule matching the contract", () => {
    const mapper = new ActuationMapper(asset);
    const s = mapper.solve(asset.sample!.hT);
    expect(s.numPistons).toBe(asset.geometry.pistonCount);
    expect(s.numSteps).toBe(asset.geometry.numSteps);
    expect(s.a.length).toBe(s.numPistons * s.numSteps);
    expect(s.dt).toBeGreaterThan(0);
    expect(s.focalTime).toBeGreaterThan(0);
    expect(s.a instanceof Float32Array).toBe(true);
  });

  it("rejects a target of the wrong length", () => {
    const mapper = new ActuationMapper(asset);
    expect(() => mapper.solve(new Float32Array(asset.pinv.cols + 1))).toThrow(/length/);
  });
});
