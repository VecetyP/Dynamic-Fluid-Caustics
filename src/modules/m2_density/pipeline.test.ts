import { describe, it, expect } from "vitest";
import { preprocessDensity } from "./index.ts";
import { InverseCausticSolver, forwardNonlinear } from "../m3_inverse/index.ts";
import { ActuationMapper, type PinvAsset } from "../m4_actuation/index.ts";
import { forwardPlayback } from "../m5_fluid/reference.ts";
import { chooseDt, type WaveParams } from "../../physics.ts";
import assetJson from "../m4_actuation/__fixtures__/pinv_small.json";

// FULL PIPELINE, HEADLESS: a synthetic "sketch" round-trips to a caustic.
//
//   intensity --M2--> density I --M3--> surface hT --M4--> schedule
//        --M5 forward sim--> surface hT' --(recover u')--> forwardNonlinear --> I'
//
// I' (the reconstructed floor irradiance from the surface the pistons actually
// produce) should resemble the input density I. This is the whole project in one
// assertion, verified without a GPU.

const asset = assetJson as unknown as PinvAsset;
const g = asset.geometry;
const N = g.n;
const DX = g.dx;
const D = 0.15;
const NREL = 1.333;

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

function drawnBlob(n: number): Float32Array {
  // A soft off-centre blob — stands in for a user stroke.
  const a = new Float32Array(n * n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const dx = x / n - 0.42;
      const dy = y / n - 0.55;
      a[y * n + x] = Math.exp(-((dx * dx + dy * dy) / (2 * 0.13 * 0.13)));
    }
  }
  return a;
}

describe("Full pipeline: sketch → caustic (headless)", () => {
  it("the reconstructed caustic resembles the input drawing", () => {
    // M2: intensity → density.
    const density = preprocessDensity(drawnBlob(N), N);
    const I = Float64Array.from(density.I);

    // M3: density → target surface.
    const { target } = new InverseCausticSolver(N).solve(I, DX, D, NREL);

    // M4: surface → piston schedule.
    const schedule = new ActuationMapper(asset).solve(target.hT);

    // M5: replay the schedule forward to the focal surface.
    const params: WaveParams = { n: N, dx: DX, depth: g.depth, gamma: g.gamma, cflSafety: 0.9 };
    const surface = forwardPlayback(
      params, chooseDt(params), g.pistonCells, schedule.a, schedule.numPistons, schedule.numSteps
    );

    // Recover the potential from the produced surface (hT = −u/[d(nRel−1)]) and
    // compute the caustic irradiance it casts.
    const k = -D * (NREL - 1);
    const uPrime = new Float64Array(N * N);
    for (let i = 0; i < uPrime.length; i++) uPrime[i] = surface[i] * k;
    const reconstructed = forwardNonlinear(uPrime, N, DX);

    // Compare to the (normalised) input density.
    const targetNorm = new Float64Array(I.length);
    for (let i = 0; i < I.length; i++) targetNorm[i] = I[i] / density.iBar;

    const c = corr(reconstructed, targetNorm);
    // With a moderate-contrast target (M2 keeps it paraxial), M3 is faithful and
    // M4+sim reconstructs hT almost exactly, so the round trip stays tight.
    expect(c).toBeGreaterThan(0.9);
  });
});
