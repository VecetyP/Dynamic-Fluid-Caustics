/**
 * Inter-module data contracts — spec §3.2.
 *
 * Every arrow in the pipeline (§2) carries exactly one typed payload. Modules
 * communicate ONLY through these; no module reaches into another's internal
 * state. On-demand payloads (M1–M4) are plain CPU arrays — small, cache-friendly,
 * trivially serialisable for golden-image regression tests. Per-frame payloads
 * (FluidState, CausticBuffer) are GPU-resident to minimise CPU<->GPU transfer.
 *
 * NOTE: The spec structs are C++ (row-major float arrays + GPU textures). Here,
 * CPU arrays are Float32Array; GPU textures are GPUTexture handles.
 */

/** Axis-aligned bounding box in canvas pixels. */
export interface AABB {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

/** M1 → M2 · RawCanvas — rasterised user sketch. */
export interface RawCanvas {
  width: number;
  height: number;
  /** RGBA8, sRGB, GPU-resident. */
  bitmap: GPUTexture;
  /** Tight bbox of drawn content. */
  inkBounds: AABB;
}

/** M2 → M3 · DensityMap — smoothed, energy-normalised target irradiance.
 *  Invariants: I[i] >= EPS > 0 (strictly positive); sum(I) == Np*Np * iBar. */
export interface DensityMap {
  /** Solver grid dim (N' per side). */
  np: number;
  /** Row-major, length np*np, single channel. */
  I: Float32Array;
  /** Uniform source level = mean(I) > 0. */
  iBar: number;
}

/** M3 → M4 · TargetHeightmap — static surface that focuses light into I. */
export interface TargetHeightmap {
  np: number;
  /** Target surface height, metres, row-major (length np*np). */
  hT: Float32Array;
  /** Focal distance surface → floor, metres. */
  d: number;
  /** Relative refractive index n2/n1. */
  nRel: number;
}

/** M4 → M5 · PistonSchedule — per-piston reversed-time amplitude timeline. */
export interface PistonSchedule {
  /** P, indexed around tank perimeter. */
  numPistons: number;
  /** T_r reversed-time samples. */
  numSteps: number;
  /** amplitude[P][T_r], row-major per piston (length numPistons*numSteps). */
  a: Float32Array;
  /** Timeline sample period (== sim dt). */
  dt: number;
  /** t at which convergence occurs. */
  focalTime: number;
}

/** M5 → M6 · FluidState (GPU-resident) — current surface + derived normals. */
export interface FluidState {
  width: number;
  height: number;
  /** R32F — current surface h^n. */
  heightTex: GPUTexture;
  /** RGBA16F — derived surface normals (xyz) + slope magnitude (w). */
  normalTex: GPUTexture;
  /** dx (== dy), metres. */
  cellSize: number;
}

/** M6 → screen · CausticBuffer — accumulated floor irradiance. */
export interface CausticBuffer {
  /** R32-uint fixed-point during accumulation, OR rgba16float when using the
   *  additive-blend splat path (path B, chosen for WebGPU). Resolved to R16F
   *  irradiance in the tone pass. */
  floorAccum: GPUTexture;
  /** Tone-map scale, auto or user-set. */
  exposure: number;
}
