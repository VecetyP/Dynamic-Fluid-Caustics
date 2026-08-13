/**
 * Shared physics constants and stability math — spec §4.2 / §6.
 *
 * Kept framework-free (no WebGPU) so it can be unit-tested under Node and reused
 * by both the M5 GPU sim and a CPU reference solver.
 */

export const GRAVITY = 9.81; // m/s^2

export interface WaveParams {
  /** Grid cells per side (sim grid N). */
  n: number;
  /** Cell size Δx == Δy, metres. */
  dx: number;
  /** Still-water depth H, metres (sets wave speed via shallow-water limit). */
  depth: number;
  /** Linear damping coefficient γ (models viscosity, spec §6). */
  gamma: number;
  /** Fraction of the CFL bound to use for Δt (spec §9 recommends ~0.9). */
  cflSafety: number;
}

/** Shallow-water wave speed c = √(g·H) — spec §4.2, Nomenclature. */
export function waveSpeed(p: WaveParams): number {
  return Math.sqrt(GRAVITY * p.depth);
}

/** Maximum stable timestep from the 2D CFL condition Δt ≤ Δx / (c√2) — eq 4.8. */
export function cflMaxDt(p: WaveParams): number {
  return p.dx / (waveSpeed(p) * Math.SQRT2);
}

/** Chosen timestep = cflSafety × the CFL bound. */
export function chooseDt(p: WaveParams): number {
  return p.cflSafety * cflMaxDt(p);
}

/**
 * Assert the chosen Δt respects CFL (eq 4.8). Explicit leapfrog DIVERGES if
 * violated (spec §9), so we refuse to run rather than produce garbage.
 */
export function assertCflStable(p: WaveParams, dt: number): void {
  const maxDt = cflMaxDt(p);
  if (!(dt > 0) || dt > maxDt) {
    throw new Error(
      `CFL violation: dt=${dt.toExponential(3)}s exceeds max stable ` +
        `dt=${maxDt.toExponential(3)}s (c=${waveSpeed(p).toFixed(3)} m/s, ` +
        `dx=${p.dx} m). Reduce dt or increase dx.`
    );
  }
}

export const DEFAULT_PARAMS: WaveParams = {
  n: 128, // spec kickoff: start at 128², bump to 256² once proven
  dx: 0.01, // 1 cm cells → 1.28 m tank
  depth: 0.05, // 5 cm shallow tank
  gamma: 0.4, // gentle damping; lower γ = a "low-viscosity" fluid (spec §9)
  cflSafety: 0.9,
};
