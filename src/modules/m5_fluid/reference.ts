/**
 * CPU reference implementation of the M5 leapfrog (eq 4.7).
 *
 * This is the "golden" oracle: the GPU WGSL shader must match it. Keeping a
 * plain-array version lets us regression-test the numerics under Node with no
 * WebGPU, exactly as the spec calls for (golden-heightmap tests, §3, §8 Phase 1/2).
 */

import { waveSpeed, type WaveParams } from "../../physics.ts";

/**
 * Forward-replay a PistonSchedule and return the surface at focal time T.
 *
 * The canonical actuation model shared by the whole pipeline (M4 basis, this CPU
 * reference, and the M5 GPU sim): each step, advance the free leapfrog, THEN
 * additively inject each piston's amplitude for that step into the new surface
 * at its cell. Because the M4 wave-basis is built from this exact operator, a
 * schedule a* = M⁺·h_t replayed here reconstructs h_t at focal time — the
 * closed-loop check in closed_loop.test.ts.
 *
 * @param pistonCells flat grid indices of the P pistons
 * @param a           amplitudes, row-major [P][T] (== PistonSchedule.a)
 */
export function forwardPlayback(
  params: WaveParams,
  dt: number,
  pistonCells: ArrayLike<number>,
  a: ArrayLike<number>,
  numPistons: number,
  numSteps: number
): Float32Array {
  const w = new WaveFieldCPU(params, dt);
  for (let step = 0; step < numSteps; step++) {
    w.step();
    for (let k = 0; k < numPistons; k++) {
      w.curr[pistonCells[k]] += a[k * numSteps + step];
    }
  }
  return w.curr;
}

export class WaveFieldCPU {
  readonly n: number;
  curr: Float32Array;
  prev: Float32Array;
  private next: Float32Array;
  private readonly c2dt2: number;
  private readonly damp: number;

  constructor(private readonly p: WaveParams, dt: number) {
    this.n = p.n;
    this.curr = new Float32Array(p.n * p.n);
    this.prev = new Float32Array(p.n * p.n);
    this.next = new Float32Array(p.n * p.n);
    const c = waveSpeed(p);
    this.c2dt2 = c * c * dt * dt;
    this.damp = 0.5 * p.gamma * dt;
  }

  private at(h: Float32Array, x: number, y: number): number {
    const m = this.n - 1;
    const cx = x < 0 ? 0 : x > m ? m : x; // clamp = reflective Neumann wall
    const cy = y < 0 ? 0 : y > m ? m : y;
    return h[cy * this.n + cx];
  }

  /** Inject a Gaussian displacement (zero initial velocity: same on curr+prev). */
  poke(cx: number, cy: number, radius: number, amp: number): void {
    for (let y = 0; y < this.n; y++) {
      for (let x = 0; x < this.n; x++) {
        const dx = x - cx;
        const dy = y - cy;
        const g = amp * Math.exp(-(dx * dx + dy * dy) / (2 * radius * radius));
        this.curr[y * this.n + x] += g;
        this.prev[y * this.n + x] += g;
      }
    }
  }

  step(): void {
    const { n, curr, prev, next, c2dt2, damp } = this;
    const invDx2 = 1 / (this.p.dx * this.p.dx);
    const inv = 1 / (1 + damp);
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const hc = curr[y * n + x];
        const hp = prev[y * n + x];
        const lap =
          (this.at(curr, x + 1, y) +
            this.at(curr, x - 1, y) +
            this.at(curr, x, y + 1) +
            this.at(curr, x, y - 1) -
            4 * hc) *
          invDx2;
        next[y * n + x] = inv * (2 * hc + (damp - 1) * hp + c2dt2 * lap);
      }
    }
    // Rotate prev <- curr <- next.
    this.prev = this.curr;
    this.curr = next;
    this.next = prev; // reuse old prev as scratch
  }

  /** Σ h² — a proxy for wave energy; must be non-increasing under damping. */
  sumSquares(): number {
    let s = 0;
    for (let i = 0; i < this.curr.length; i++) s += this.curr[i] * this.curr[i];
    return s;
  }
}
