import { describe, it, expect } from "vitest";
import { ActuationMapper, type PinvAsset } from "./index.ts";
import { forwardPlayback } from "../m5_fluid/reference.ts";
import { chooseDt, type WaveParams } from "../../physics.ts";
import assetJson from "./__fixtures__/pinv_small.json";

// PHASE-3 HEADLESS ACCEPTANCE: the full inverse→forward loop, no GPU.
//
//   target surface h_t  --M4-->  PistonSchedule a*  --M5 forward sim-->  surface(T)
//
// If the actuation model is consistent (M4 basis built from the same injection
// rule the forward sim replays), surface(T) reconstructs h_t. This closes the
// loop that the real app runs on the GPU, and pins the CPU/GPU sim to the exact
// operator M4 assumes.

const asset = assetJson as PinvAsset;

function relL2(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let num = 0, den = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    num += d * d;
    den += b[i] * b[i];
  }
  return Math.sqrt(num / den);
}

function corr(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = a.length;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - ma, db = b[i] - mb;
    sab += da * db; saa += da * da; sbb += db * db;
  }
  return sab / Math.sqrt(saa * sbb);
}

describe("Phase 3 closed loop: target → M4 schedule → M5 forward sim → target", () => {
  const g = asset.geometry;
  const params: WaveParams = {
    n: g.n,
    dx: g.dx,
    depth: g.depth,
    gamma: g.gamma,
    cflSafety: 0.9,
  };

  it("TS derives the same dt as the Python export (CFL formula match)", () => {
    expect(Math.abs(chooseDt(params) - g.dt)).toBeLessThan(1e-9);
  });

  it("replaying the recovered schedule reconstructs the target surface", () => {
    const dt = chooseDt(params);
    const target = asset.sample!.hT;

    // Full loop: solve the actuation from the target, then replay it forward.
    const schedule = new ActuationMapper(asset).solve(target);
    const surface = forwardPlayback(
      params, dt, g.pistonCells, schedule.a, schedule.numPistons, schedule.numSteps
    );

    const err = relL2(surface, target);
    const c = corr(surface, target);
    // Basis-limited reconstruction (~7% at this tiny geometry) + f32 drift.
    expect(err).toBeLessThan(0.12);
    expect(c).toBeGreaterThan(0.95);
  });

  it("surface is quiescent well before focal time (transient focus)", () => {
    // Sanity on the 'transient focus' property (spec §6): at an early time the
    // surface has not yet converged, so it should correlate far less with h_t.
    const dt = chooseDt(params);
    const target = asset.sample!.hT;
    const schedule = new ActuationMapper(asset).solve(target);

    const early = forwardPlayback(
      params, dt, g.pistonCells, schedule.a, schedule.numPistons,
      Math.max(1, Math.floor(schedule.numSteps / 3))
    );
    const full = forwardPlayback(
      params, dt, g.pistonCells, schedule.a, schedule.numPistons, schedule.numSteps
    );
    expect(corr(full, target)).toBeGreaterThan(corr(early, target));
  });
});
