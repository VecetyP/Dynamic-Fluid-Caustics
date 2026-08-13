/**
 * M-B · CpuWaterPlayer — drives the 3D water surface from the CPU wave sim.
 *
 * This replays a PistonSchedule on the CPU using the SAME operator the whole
 * pipeline shares (advance the leapfrog, THEN additively inject each piston's
 * amplitude into the new surface at its cell — see `m5_fluid/reference.ts`), and
 * exposes the live height field so the Three.js water mesh can be displaced each
 * frame. Running the water on the CPU sidesteps all WebGPU↔WebGL interop.
 *
 * It mirrors M7's pulse/hold/loop pacing: spread the T build-up steps over more
 * frames so convergence is watchable, hold the focal surface, then reset and
 * replay so the caustic pulses steadily (spec §9 Pulse Mode).
 *
 * VERIFICATION: because `advanceStep` applies the exact `forwardPlayback` loop
 * body, running the player through all `numSteps` steps reproduces
 * `forwardPlayback(...)` bit-for-bit — asserted in `water_player.test.ts`, so the
 * 3D surface is provably the verified physics (no GPU needed to check it).
 */

import { WaveFieldCPU } from "../m5_fluid/reference.ts";
import type { WaveParams } from "../../physics.ts";
import type { PistonSchedule } from "../../contracts/index.ts";

// Pulse pacing — matches M7 (m7_orchestrator) so the 3D water and the 2D caustic
// preview build up and hold in step.
const FRAMES_PER_STEP = 3;
const HOLD_FRAMES = 150;

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

  private frameCounter = 0;
  private holdCounter = 0;
  private speed = 1;

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
    this.phase = "building";
  }

  /** Zero the surface and rewind the playback cursor (replay from rest). */
  reset(): void {
    if (!this.params) return;
    this.w = new WaveFieldCPU(this.params, this.dt);
    this.cursor = 0;
    this.frameCounter = 0;
    this.holdCounter = 0;
    this.phase = this.numSteps > 0 ? "building" : "idle";
  }

  setSpeed(s: number): void {
    this.speed = Math.max(0.1, Math.min(8, s));
  }

  private framesPerStep(): number {
    return Math.max(1, Math.round(FRAMES_PER_STEP / this.speed));
  }
  private holdFrames(): number {
    return Math.max(1, Math.round(HOLD_FRAMES / this.speed));
  }

  /**
   * Advance the sim by exactly ONE physics step at the current cursor: free
   * leapfrog, then inject this step's piston amplitudes. This is the canonical
   * operator (identical to `forwardPlayback`'s loop body). Returns false when
   * there is nothing left to build.
   */
  advanceStep(): boolean {
    if (!this.w || this.phase !== "building") return false;
    this.w.step();
    for (let k = 0; k < this.numPistons; k++) {
      this.w.curr[this.pistonCells[k]] += this.a[k * this.numSteps + this.cursor];
    }
    this.cursor++;
    if (this.cursor >= this.numSteps) this.phase = "hold";
    return true;
  }

  /**
   * One animation frame: pace the build-up, hold the focal surface, then reset
   * and re-pulse. Call once per rendered frame. No-op until a schedule is loaded.
   */
  tick(): void {
    if (this.phase === "idle" || !this.w) return;

    if (this.phase === "building") {
      if (this.frameCounter % this.framesPerStep() === 0) this.advanceStep();
      this.frameCounter++;
    } else {
      // hold
      this.holdCounter++;
      if (this.holdCounter >= this.holdFrames()) this.reset();
    }
  }

  /** Current surface height field (row-major n·n), or null before load. */
  height(): Float32Array | null {
    return this.w?.curr ?? null;
  }

  /** Number of pistons in the loaded schedule. */
  get pistonCount(): number {
    return this.numPistons;
  }

  /** The schedule step whose injection is currently showing on the surface:
   *  the most recently applied step (cursor-1), clamped; -1 before any step. */
  displayStep(): number {
    if (this.cursor <= 0 || this.numSteps === 0) return -1;
    return Math.min(this.cursor - 1, this.numSteps - 1);
  }

  /** Per-piston amplitude injected at the current display step (length P). This
   *  is what a wall piston is "doing" right now, so it drives the piston meshes
   *  in lockstep with the water. Fills `out` if provided. Zeros before load. */
  pistonAmplitudes(out?: Float32Array): Float32Array {
    const P = this.numPistons;
    const res = out && out.length >= P ? out : new Float32Array(P);
    const s = this.displayStep();
    for (let k = 0; k < P; k++) {
      res[k] = s >= 0 ? this.a[k * this.numSteps + s] : 0;
    }
    return res;
  }

  /** Largest |amplitude| anywhere in the loaded schedule (for display scaling). */
  maxAbsAmplitude(): number {
    let m = 0;
    for (let i = 0; i < this.a.length; i++) m = Math.max(m, Math.abs(this.a[i]));
    return m;
  }
}
