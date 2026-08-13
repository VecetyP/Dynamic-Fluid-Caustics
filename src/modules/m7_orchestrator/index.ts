/**
 * M7 · Orchestration State Machine (Phase-1 subset).
 *
 * The full spec drives on-demand solves (M1–M4) and Pulse-Mode looping. For
 * Phase 1 we only need the per-frame path: advance M5, render M6, forward user
 * pokes. States are stubbed so later phases slot in without reshaping the loop.
 */

import type { GpuContext } from "../../gpu/device.ts";
import { FluidSim } from "../m5_fluid/index.ts";
import { CausticRenderer, type RenderOptions } from "../m6_render/index.ts";
import { DEFAULT_PARAMS, type WaveParams } from "../../physics.ts";

export type SimState = "IDLE" | "SOLVING" | "PLAYBACK" | "PULSE";

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

export class Orchestrator {
  state: SimState = "PLAYBACK";
  readonly sim: FluidSim;
  readonly renderer: CausticRenderer;
  private readonly gpu: GpuContext;

  constructor(gpu: GpuContext, cfg: OrchestratorConfig = {}) {
    this.gpu = gpu;
    const wave: WaveParams = { ...DEFAULT_PARAMS, ...cfg.wave };
    const render: RenderOptions = { ...DEFAULT_RENDER, ...cfg.render };

    this.sim = new FluidSim(gpu.device, wave);
    this.renderer = new CausticRenderer(gpu.device, wave.n, wave.dx, gpu.format, render);
  }

  /** Poke from normalised canvas coords (0..1). */
  pokeNormalised(u: number, v: number, amplitude = 0.5): void {
    const n = this.sim.params.n;
    this.sim.poke({
      x: u * n,
      y: v * n,
      radius: Math.max(1.5, n * 0.02),
      amplitude,
    });
  }

  /** One tick: advance the fluid, render the caustic to the swap-chain. */
  frame(): void {
    if (this.state === "IDLE") return;
    const encoder = this.gpu.device.createCommandEncoder({ label: "frame" });
    this.sim.encode(encoder);
    const swapView = this.gpu.context.getCurrentTexture().createView();
    this.renderer.encode(encoder, this.sim.state, swapView);
    this.gpu.device.queue.submit([encoder.finish()]);
  }

  setExposure(e: number): void {
    this.renderer.setExposure(e);
  }

  destroy(): void {
    this.sim.destroy();
    this.renderer.destroy();
  }
}
