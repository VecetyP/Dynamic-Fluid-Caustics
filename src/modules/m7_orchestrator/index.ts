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

// Pulse pacing (visual only — physics still advances exactly one step per
// injected cursor). Spread the T build-up steps over more frames so the
// convergence is watchable, then hold the focal surface so the caustic is a
// steady image instead of a 60 Hz strobe.
const FRAMES_PER_STEP = 3; // render this many frames per physics step during build-up
const HOLD_FRAMES = 150; // ~2.5 s holding the focused caustic before re-pulsing

export class Orchestrator {
  state: SimState = "INTERACTIVE";
  readonly sim: FluidSim;
  readonly renderer: CausticRenderer;
  private readonly gpu: GpuContext;
  private cursor = 0;
  private phase: "building" | "hold" = "building";
  private frameCounter = 0;
  private holdCounter = 0;

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
    this.frameCounter = 0;
    this.holdCounter = 0;
    this.state = "PULSE";
  }

  /** Fraction through the current pulse cycle (0..1) — for UI / diagnostics. */
  get pulsePhase(): number {
    const t = this.sim.numSteps;
    return t > 0 ? this.cursor / t : 0;
  }

  /** One tick: advance the fluid (when due) and render the caustic. */
  frame(): void {
    if (this.state === "IDLE") return;

    // Decide whether to advance the physics this frame, and with what injection.
    let doStep = true;
    let pistonStep: number | null = null;
    if (this.state === "PULSE") {
      if (this.phase === "hold") {
        doStep = false; // freeze the focal surface → steady caustic
      } else {
        doStep = this.frameCounter % FRAMES_PER_STEP === 0; // pace the build-up
        pistonStep = doStep ? this.cursor : null;
      }
    }

    const encoder = this.gpu.device.createCommandEncoder({ label: "frame" });
    if (doStep) this.sim.encode(encoder, pistonStep);
    const swapView = this.gpu.context.getCurrentTexture().createView();
    this.renderer.encode(encoder, this.sim.state, swapView);
    this.gpu.device.queue.submit([encoder.finish()]);

    this.frameCounter++;
    if (this.state !== "PULSE") return;

    if (this.phase === "building") {
      if (doStep) {
        this.cursor++;
        if (this.cursor >= this.sim.numSteps) this.phase = "hold"; // focal reached
      }
    } else {
      // Holding the focused image; after a beat, reset and re-pulse (spec §9).
      this.holdCounter++;
      if (this.holdCounter >= HOLD_FRAMES) {
        this.sim.reset();
        this.cursor = 0;
        this.holdCounter = 0;
        this.phase = "building";
      }
    }
  }

  setExposure(e: number): void {
    this.renderer.setExposure(e);
  }

  destroy(): void {
    this.sim.destroy();
    this.renderer.destroy();
  }
}
