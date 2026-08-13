/**
 * M4 · Wave Actuation Mapper (runtime).
 *
 * The heavy step — assembling the wave-basis matrix M and its regularised
 * pseudoinverse M⁺ = (MᵀM + λI)⁻¹Mᵀ (spec §4.3, eq 4.12) — is geometry-only and
 * done offline (see `prototypes/m4_actuation/export_pinv.py`). At runtime this
 * module just loads that shipped M⁺ and turns a target surface into a piston
 * timeline with a single matrix-vector product:
 *
 *     a = M⁺ · h_t        (PistonSchedule)
 *
 * The NumPy prototype validated the math; `m4.test.ts` checks this TS matvec
 * reproduces the NumPy a* exactly for the exported golden sample.
 */

import type { PistonSchedule, TargetHeightmap } from "../../contracts/index.ts";

/** Shape of the JSON asset produced by export_pinv.py. */
export interface PinvAsset {
  geometry: {
    n: number;
    dx: number;
    depth: number;
    gamma: number;
    pistonCount: number;
    numSteps: number;
    lambda: number;
    dt: number;
    focalTime: number;
    pistonCells: number[];
  };
  pinv: {
    rows: number; // P·T
    cols: number; // N (surface samples)
    data: number[]; // row-major, length rows*cols
  };
  sample?: { hT: number[]; aExpected: number[] };
}

export class ActuationMapper {
  readonly rows: number; // P·T
  readonly cols: number; // N
  readonly numPistons: number;
  readonly numSteps: number;
  readonly dt: number;
  readonly focalTime: number;
  private readonly pinv: Float64Array;

  constructor(asset: PinvAsset) {
    const { rows, cols, data } = asset.pinv;
    if (data.length !== rows * cols) {
      throw new Error(`pinv data length ${data.length} != rows*cols ${rows * cols}`);
    }
    if (rows !== asset.geometry.pistonCount * asset.geometry.numSteps) {
      throw new Error("pinv rows must equal pistonCount * numSteps");
    }
    this.rows = rows;
    this.cols = cols;
    this.numPistons = asset.geometry.pistonCount;
    this.numSteps = asset.geometry.numSteps;
    this.dt = asset.geometry.dt;
    this.focalTime = asset.geometry.focalTime;
    this.pinv = Float64Array.from(data);
  }

  /**
   * Map a flattened target surface (row-major, length N) to a PistonSchedule.
   * Accepts a raw Float32Array/number[] or a TargetHeightmap (uses its hT).
   */
  solve(target: Float32Array | number[] | TargetHeightmap): PistonSchedule {
    const hT: ArrayLike<number> =
      target instanceof Float32Array || Array.isArray(target) ? target : target.hT;
    if (hT.length !== this.cols) {
      throw new Error(`target length ${hT.length} != solver cols ${this.cols}`);
    }

    // a[i] = Σ_j M⁺[i, j] · h_t[j]   (dense matvec, rows·cols mults)
    const a = new Float32Array(this.rows);
    const P = this.pinv;
    const cols = this.cols;
    for (let i = 0; i < this.rows; i++) {
      let s = 0;
      const base = i * cols;
      for (let j = 0; j < cols; j++) s += P[base + j] * hT[j];
      a[i] = s;
    }

    // Rows are ordered (piston-major, time-minor) == amplitude[P][T_r], so `a`
    // already matches the PistonSchedule layout directly.
    return {
      numPistons: this.numPistons,
      numSteps: this.numSteps,
      a,
      dt: this.dt,
      focalTime: this.focalTime,
    };
  }
}
