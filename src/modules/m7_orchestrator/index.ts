/**
 * M7 · Orchestration State Machine.
 *
 * Drives the per-frame path and the Phase-3 playback loop.
 *   - INTERACTIVE: free-running ripple; pointer pokes the surface (Phase 1).
 *   - PULSE: replay a PistonSchedule from rest; at focal time T the surface
 *     converges to the target and the caustic forms, then reset and replay so the
 *     image pulses steadily (spec §9 Pulse Mode).
 */

import type { GpuContext } from "../../gpu/device.ts";
import type { PistonSchedule } from "../../contracts/index.ts";
import { FluidSim } from "../m5_fluid/index.ts";
import { CausticRenderer, type RenderOptions } from "../m6_render/index.ts";
import { DEFAULT_PARAMS, type WaveParams } from "../../physics.ts";
import { stepSeconds, holdSeconds } from "../../playback_timing.ts";

export type SimState = "IDLE" | "INTERACTIVE" | "PULSE";

export interface OrchestratorConfig {
  wave?: Partial<WaveParams>;
  render?: Partial<RenderOptions>;
}

const DEFAULT_RENDER: RenderOptions = {
  focalDistance: 0.15, // 15 cm surface → floor
  nRel: 1.333, // water
  cellEnergy: 0.03,
  exposure: 1.5,
};

// Pulse pacing is WALL-CLOCK (seconds), shared with the 3D water player via
// playback_timing.ts so the 2D caustic preview and the 3D view stay in step.
// Physics still advances exactly one step per injected cursor; we just pace the
// steps by elapsed time and hold the focal surface as a steady image.

export class Orchestrator {
  state: SimState = "INTERACTIVE";
  readonly sim: FluidSim;
  readonly renderer: CausticRenderer;
  private readonly gpu: GpuContext;
  private cursor = 0;
  private phase: "building" | "hold" = "building";
  private accum = 0; // seconds accumulated toward the next step / hold end
  private speed = 1; // playback speed multiplier (slider-controlled)

  constructor(gpu: GpuContext, cfg: OrchestratorConfig = {}) {
    this.gpu = gpu;
    const wave: WaveParams = { ...DEFAULT_PARAMS, ...cfg.wave };
    const render: RenderOptions = { ...DEFAULT_RENDER, ...cfg.render };

    this.sim = new FluidSim(gpu.device, wave);
    this.renderer = new CausticRenderer(gpu.device, wave.n, wave.dx, gpu.format, render);
  }

  /** Poke from normalised canvas coords (0..1). Interactive mode only. */
  pokeNormalised(u: number, v: number, amplitude = 0.5): void {
    const n = this.sim.params.n;
    this.sim.poke({
      x: u * n,
      y: v * n,
      radius: Math.max(1.5, n * 0.02),
      amplitude,
    });
  }

  /** Load a schedule and begin pulsing it from rest. */
  startPulse(schedule: PistonSchedule, pistonCells: Uint32Array): void {
    this.sim.loadSchedule(schedule, pistonCells);
    this.sim.reset();
    this.cursor = 0;
    this.phase = "building";
    this.accum = 0;
    this.state = "PULSE";
  }

  /** Fraction through the current pulse cycle (0..1) — for UI / diagnostics. */
  get pulsePhase(): number {
    const t = this.sim.numSteps;
    return t > 0 ? this.cursor / t : 0;
  }

  /** One tick: advance the fluid (when due, paced by `dt` seconds) and render the
   *  caustic every frame. `dt` is real elapsed seconds since the last frame. */
  frame(dt = 1 / 60): void {
    if (this.state === "IDLE") return;

    // Decide whether to advance the physics this frame, and with what injection.
    // This pacing MIRRORS the 3D water player (water_player.ts) EXACTLY so the two
    // views stay phase-locked: step while cursor < numSteps, and only transition to
    // hold one step-duration AFTER the last step (the player needs that extra sd to
    // display its final interpolation segment — matching it here is what stops the
    // slow, speed-dependent drift between the 2D preview and the 3D floor).
    let doStep = false;
    let pistonStep: number | null = null;

    if (this.state === "PULSE") {
      if (this.phase === "building") {
        const sd = stepSeconds(this.sim.numSteps, this.speed);
        this.accum += dt;
        if (this.accum >= sd) {
          this.accum -= sd; // carry the remainder exactly (like the player)
          if (this.cursor < this.sim.numSteps) {
            doStep = true;
            pistonStep = this.cursor;
            this.cursor++;
          } else {
            this.phase = "hold"; // one sd past the last step → focal settled
            this.accum = 0;
          }
        }
      } else {
        // Holding the focused image; after a beat, reset and re-pulse (spec §9).
        this.accum += dt;
        if (this.accum >= holdSeconds(this.speed)) {
          this.sim.reset();
          this.cursor = 0;
          this.accum = 0;
          this.phase = "building";
        }
      }
    } else {
      doStep = true; // INTERACTIVE free-run: advance every frame
    }

    const encoder = this.gpu.device.createCommandEncoder({ label: "frame" });
    if (doStep) this.sim.encode(encoder, pistonStep);
    const swapView = this.gpu.context.getCurrentTexture().createView();
    this.renderer.encode(encoder, this.sim.state, swapView);
    this.gpu.device.queue.submit([encoder.finish()]);
  }

  setExposure(e: number): void {
    this.renderer.setExposure(e);
  }

  /** Playback speed multiplier (1 = default). Higher = faster build-up + hold. */
  setSpeed(s: number): void {
    this.speed = Math.max(0.1, Math.min(8, s));
  }

  destroy(): void {
    this.sim.destroy();
    this.renderer.destroy();
  }
}
