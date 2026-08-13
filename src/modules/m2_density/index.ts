/**
 * M2 · Density Map Preprocessor (spec Stage 1 / §3.1).
 *
 * Turns a raw greyscale intensity field (from the M1 sketch, downsampled to the
 * solver grid) into a DensityMap the inverse solver can use:
 *   - Gaussian blur to band-limit — remove spatial frequencies the fluid surface
 *     cannot physically form (spec §6; high frequencies also decay fastest under
 *     viscosity, so demanding them just adds error).
 *   - a positive ambient floor so the target irradiance is STRICTLY POSITIVE
 *     (spec invariant I[i] >= EPS > 0): the whole floor is lit, the drawing is
 *     the bright part on top.
 *   - iBar = mean(I) is the uniform source level; M3's RHS (1 − I/Ī) is then
 *     automatically zero-mean (Neumann compatibility).
 *
 * Framework-free (no WebGPU / no DOM) so it unit-tests under Node.
 */

import type { DensityMap } from "../../contracts/index.ts";

export interface DensityOptions {
  /** Gaussian blur sigma, in grid cells. Larger ⇒ softer, more physical. */
  blurSigma: number;
  /** Ambient light floor added everywhere (keeps I strictly positive). */
  ambient: number;
  /** Weight of the drawing on top of the ambient base. Keeping this modest
   *  keeps target CONTRAST low so the paraxial inverse solver (M3) stays in its
   *  valid regime — high contrast breaks the linearisation (spec §4.1.2). */
  gain: number;
}

export const DEFAULT_DENSITY_OPTIONS: DensityOptions = {
  blurSigma: 1.5,
  ambient: 1.0,
  gain: 0.4,
};

/** Separable Gaussian blur with clamped (edge-replicated) borders. */
export function gaussianBlur(src: Float32Array, n: number, sigma: number): Float32Array {
  if (sigma <= 0) return Float32Array.from(src);
  const radius = Math.max(1, Math.ceil(3 * sigma));
  const kernel = new Float64Array(2 * radius + 1);
  let ksum = 0;
  for (let i = -radius; i <= radius; i++) {
    const w = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel[i + radius] = w;
    ksum += w;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= ksum;

  const clamp = (v: number) => (v < 0 ? 0 : v > n - 1 ? n - 1 : v);
  const tmp = new Float32Array(n * n);
  // Horizontal.
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      let s = 0;
      for (let k = -radius; k <= radius; k++) s += kernel[k + radius] * src[y * n + clamp(x + k)];
      tmp[y * n + x] = s;
    }
  }
  // Vertical.
  const out = new Float32Array(n * n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      let s = 0;
      for (let k = -radius; k <= radius; k++) s += kernel[k + radius] * tmp[clamp(y + k) * n + x];
      out[y * n + x] = s;
    }
  }
  return out;
}

/**
 * @param intensity greyscale in [0,1], row-major length n*n (bright = drawn).
 * @param n         solver grid dim (must match M3/M4 geometry).
 */
export function preprocessDensity(
  intensity: Float32Array,
  n: number,
  opts: DensityOptions = DEFAULT_DENSITY_OPTIONS
): DensityMap {
  if (intensity.length !== n * n) throw new Error(`intensity must be length ${n * n}`);

  const blurred = gaussianBlur(intensity, n, opts.blurSigma);

  const I = new Float32Array(n * n);
  let sum = 0;
  for (let i = 0; i < I.length; i++) {
    // Strictly positive (ambient > 0); the drawing is a modest bump on top.
    const v = opts.ambient + opts.gain * blurred[i];
    I[i] = v;
    sum += v;
  }
  return { np: n, I, iBar: sum / I.length };
}
