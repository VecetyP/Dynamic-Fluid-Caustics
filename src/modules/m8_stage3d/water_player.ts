/**
 * M-B/M-C · CpuWaterPlayer — drives the 3D water + pistons from the CPU wave sim.
 *
 * Replays a PistonSchedule on the CPU using the SAME operator the whole pipeline
 * shares (advance the leapfrog, THEN additively inject each piston's amplitude
 * into the new surface at its cell — see `m5_fluid/reference.ts`). Running the
 * water on the CPU sidesteps all WebGPU↔WebGL interop.
 *
 * TWO LAYERS:
 *  - Discrete "physics truth" (`advanceStep`, `height`, `pistonAmplitudes`,
 *    `displayStep`) — exact integer steps; used by the headless tests, which pin
 *    the player to the verified `forwardPlayback` oracle.
 *  - Smooth playback (`tick(dt)`, `renderHeight`, `renderPistons`) — WALL-CLOCK
 *    timed and LINEARLY INTERPOLATED between successive physics steps, so the
 *    surface and paddles morph continuously instead of stepping. This is what the
 *    3D view uses: frame-rate independent (no jitter on fast displays) and smooth
 *    even at slow playback speeds. Pace: BASE_STEP_SECONDS per physics step,
 *    scaled by `speed`; hold the focal surface, then reset and re-pulse (spec §9).
 *
 * VERIFICATION (`water_player.test.ts`): running the discrete path to focal
 * reproduces `forwardPlayback` bit-for-bit, and the smooth path lands exactly on
 * that same focal surface at frac=1 — so the 3D playback is the verified physics.
 */

import { WaveFieldCPU } from "../m5_fluid/reference.ts";
import type { WaveParams } from "../../physics.ts";
import type { PistonSchedule } from "../../contracts/index.ts";
import { stepSeconds, holdSeconds } from "../../playback_timing.ts";

export type WaterPhase = "idle" | "building" | "hold";

export class CpuWaterPlayer {
  phase: WaterPhase = "idle";
  cursor = 0;

  private w: WaveFieldCPU | null = null;
  private params: WaveParams | null = null;
  private dt = 0;
  private pistonCells: ArrayLike<number> = [];
  private a: ArrayLike<number> = [];
  private numPistons = 0;
  private numSteps = 0;
  private speed = 1;

  // Smooth-playback state: interpolate segFrom → segTo over one step's duration.
  private segFrom: Float32Array = new Float32Array(0);
  private segTo: Float32Array = new Float32Array(0);
  private segFromCol = -1; // piston column that produced segFrom (-1 = rest)
  private segToCol = -1; //   piston column that produced segTo
  private accum = 0; // seconds elapsed within the current segment / hold

  /** Grid size of the height field (n per side); 0 until a schedule is loaded. */
  get n(): number {
    return this.params?.n ?? 0;
  }

  /** Load a schedule and begin pulsing it from rest. */
  load(schedule: PistonSchedule, pistonCells: ArrayLike<number>, params: WaveParams): void {
    this.params = params;
    this.dt = schedule.dt;
    this.pistonCells = pistonCells;
    this.a = schedule.a;
    this.numPistons = schedule.numPistons;
    this.numSteps = schedule.numSteps;
    this.reset();
  }

  /** Zero the surface and rewind to rest (replay from the start). */
  reset(): void {
    if (!this.params) return;
    this.w = new WaveFieldCPU(this.params, this.dt);
    this.cursor = 0;
    this.accum = 0;
    this.phase = this.numSteps > 0 ? "building" : "idle";
    const N = this.params.n * this.params.n;
    this.segFrom = new Float32Array(N); // rest
    this.segTo = new Float32Array(N); // rest
    this.segFromCol = -1;
    this.segToCol = -1;
  }

  setSpeed(s: number): void {
    this.speed = Math.max(0.1, Math.min(8, s));
  }

  private stepDuration(): number {
    return stepSeconds(this.numSteps, this.speed);
  }
  private holdDuration(): number {
    return holdSeconds(this.speed);
  }

  // ---------------------------------------------------------------------------
  // Discrete physics truth (exact integer steps) — used by tests
  // ---------------------------------------------------------------------------

  /** Advance the sim by exactly one physics step: free leapfrog, then inject this
   *  step's piston amplitudes. Returns the applied column, or -1 if none left. */
  private applyStep(): number {
    if (!this.w || this.cursor >= this.numSteps) return -1;
    this.w.step();
    const T = this.numSteps;
    for (let k = 0; k < this.numPistons; k++) {
      this.w.curr[this.pistonCells[k]] += this.a[k * T + this.cursor];
    }
    return this.cursor++;
  }

  /** Discrete single step (canonical operator, == `forwardPlayback` loop body).
   *  Returns false when the build-up is complete. */
  advanceStep(): boolean {
    if (!this.w || this.phase !== "building") return false;
    const col = this.applyStep();
    if (col < 0) {
      this.phase = "hold";
      return false;
    }
    if (this.cursor >= this.numSteps) this.phase = "hold";
    return true;
  }

  /** Current discrete surface (row-major n·n), or null before load. */
  height(): Float32Array | null {
    return this.w?.curr ?? null;
  }

  get pistonCount(): number {
    return this.numPistons;
  }

  /** Most-recently-injected step (cursor-1, clamped); -1 before any step. */
  displayStep(): number {
    if (this.cursor <= 0 || this.numSteps === 0) return -1;
    return Math.min(this.cursor - 1, this.numSteps - 1);
  }

  /** Per-piston amplitude at the current DISCRETE display step (length P). */
  pistonAmplitudes(out?: Float32Array): Float32Array {
    const P = this.numPistons;
    const res = out && out.length >= P ? out : new Float32Array(P);
    const s = this.displayStep();
    for (let k = 0; k < P; k++) res[k] = s >= 0 ? this.a[k * this.numSteps + s] : 0;
    return res;
  }

  /** Largest |amplitude| anywhere in the loaded schedule (for display scaling). */
  maxAbsAmplitude(): number {
    let m = 0;
    for (let i = 0; i < this.a.length; i++) m = Math.max(m, Math.abs(this.a[i]));
    return m;
  }

  // ---------------------------------------------------------------------------
  // Smooth playback (wall-clock + interpolation) — used by the 3D view
  // ---------------------------------------------------------------------------

  /** Take one physics step and roll the interpolation snapshots so the next
   *  segment morphs segFrom(step i) → segTo(step i+1). */
  private performStep(): void {
    if (!this.w) return;
    this.segFrom.set(this.w.curr);
    this.segFromCol = this.cursor > 0 ? this.cursor - 1 : -1;
    const col = this.applyStep();
    this.segTo.set(this.w.curr);
    this.segToCol = col;
  }

  /** Fraction through the current segment (0..1); 1 while holding. */
  private frac(): number {
    if (this.phase !== "building") return 1;
    const sd = this.stepDuration();
    return sd > 0 ? Math.min(1, this.accum / sd) : 1;
  }

  /** Advance wall-clock playback by `dt` seconds. Paces steps, holds the focal
   *  surface, then resets and re-pulses. No-op until a schedule is loaded. */
  tick(dt: number): void {
    if (this.phase === "idle" || !this.w) return;

    if (this.phase === "building") {
      this.accum += dt;
      const sd = this.stepDuration();
      let guard = 0;
      while (this.phase === "building" && this.accum >= sd && guard++ < 10_000) {
        if (this.cursor < this.numSteps) {
          this.accum -= sd;
          this.performStep();
        } else {
          this.phase = "hold"; // finished the final segment
          this.accum = 0;
        }
      }
    } else {
      // hold the focused surface, then re-pulse
      this.accum += dt;
      if (this.accum >= this.holdDuration()) this.reset();
    }
  }

  /** Interpolated surface height field for rendering (row-major n·n). Fills `out`
   *  if provided. Null before load. */
  renderHeight(out?: Float32Array): Float32Array | null {
    if (!this.w) return null;
    const N = this.segTo.length;
    const res = out && out.length >= N ? out : new Float32Array(N);
    const f = this.frac();
    const from = this.segFrom;
    const to = this.segTo;
    for (let i = 0; i < N; i++) res[i] = from[i] + (to[i] - from[i]) * f;
    return res;
  }

  /** Interpolated per-piston amplitude for rendering (length P). Fills `out`. */
  renderPistons(out?: Float32Array): Float32Array {
    const P = this.numPistons;
    const res = out && out.length >= P ? out : new Float32Array(P);
    const T = this.numSteps;
    const f = this.frac();
    for (let k = 0; k < P; k++) {
      const af = this.segFromCol < 0 ? 0 : this.a[k * T + this.segFromCol];
      const at = this.segToCol < 0 ? 0 : this.a[k * T + this.segToCol];
      res[k] = af + (at - af) * f;
    }
    return res;
  }
}
