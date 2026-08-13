/**
 * M5 · Fluid Simulation Engine.
 *
 * Advances the damped wave field one step per frame (eq 4.7) on a ping-pong of
 * r32float textures, and exposes the current surface as the height half of a
 * FluidState. Normal derivation lives in M6 (it consumes this height texture).
 *
 * Two excitation paths share the same leapfrog:
 *   - interactive "poke" (uniform Gaussian, injected in the wave step), and
 *   - PistonSchedule playback (Phase 3): a scatter pass writes each step's piston
 *     amplitudes into an injection texture that the wave step adds into h^{n+1} —
 *     the exact rule the M4 basis was built from (see reference.ts::forwardPlayback).
 */

import type { FluidState, PistonSchedule } from "../../contracts/index.ts";
import { createHeightPingPong, type PingPong } from "../../gpu/pingpong.ts";
import { createComputePipeline, createStorageTexture, dispatchCount } from "../../gpu/helpers.ts";
import { assertCflStable, chooseDt, waveSpeed, type WaveParams } from "../../physics.ts";
import waveStepWgsl from "./wave_step.wgsl?raw";
import scatterWgsl from "./inject_scatter.wgsl?raw";
import clearWgsl from "./clear.wgsl?raw";

const WG = 16; // workgroup_size in the wave/clear shaders
const WG_LINE = 64; // workgroup_size in the scatter shader
const PARAM_FLOATS = 8; // SimParams: n + 7 floats, laid out as 8×4 bytes

export interface PendingPoke {
  x: number; // grid coords
  y: number;
  radius: number; // cells
  amplitude: number; // metres
}

export class FluidSim {
  readonly params: WaveParams;
  readonly dt: number;
  private readonly device: GPUDevice;
  private readonly ring: PingPong;
  private readonly pipeline: GPUComputePipeline;
  private readonly uniform: GPUBuffer;
  private readonly paramData = new ArrayBuffer(PARAM_FLOATS * 4);
  private readonly normalTex: GPUTexture;
  private queuedPoke: PendingPoke | null = null;
  step = 0;

  // --- Piston playback (Phase 3) ---
  private readonly injTex: GPUTexture;
  private readonly scatterPipeline: GPUComputePipeline;
  private readonly scatterUniform: GPUBuffer;
  private readonly clearPipeline: GPUComputePipeline;
  private readonly clearUniform: GPUBuffer;
  private cellsBuf: GPUBuffer | null = null;
  private ampsBuf: GPUBuffer | null = null;
  private schedP = 0;
  private schedT = 0;
  private scheduleLoaded = false;

  constructor(device: GPUDevice, params: WaveParams) {
    this.device = device;
    this.params = params;
    this.dt = chooseDt(params);
    // Fail loudly at init rather than diverge mid-run (spec §9).
    assertCflStable(params, this.dt);

    this.ring = createHeightPingPong(device, params.n);
    this.pipeline = createComputePipeline(device, waveStepWgsl, "m5.wave_step");
    this.uniform = device.createBuffer({
      label: "m5.SimParams",
      size: PARAM_FLOATS * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // RGBA16F normals (xyz + slope mag), written by M6's normal pass.
    this.normalTex = createStorageTexture(device, params.n, "rgba16float", "m5.normals");

    // Injection texture: zero-initialised r32float, filled at piston cells by the
    // scatter pass; read (and added) by the wave step. Zero ⇒ no forcing.
    this.injTex = createStorageTexture(device, params.n, "r32float", "m5.inject");
    this.scatterPipeline = createComputePipeline(device, scatterWgsl, "m5.inject_scatter");
    this.clearPipeline = createComputePipeline(device, clearWgsl, "m5.clear");
    this.scatterUniform = device.createBuffer({
      label: "m5.ScatterParams",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.clearUniform = device.createBuffer({
      label: "m5.ClearParams",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const cu = new ArrayBuffer(16);
    new DataView(cu).setUint32(0, params.n, true);
    device.queue.writeBuffer(this.clearUniform, 0, cu);
  }

  /** Queue a Gaussian displacement to be injected on the next step. */
  poke(p: PendingPoke): void {
    this.queuedPoke = p;
  }

  /** Load a PistonSchedule + its piston cell layout for playback. */
  loadSchedule(schedule: PistonSchedule, pistonCells: Uint32Array): void {
    if (pistonCells.length !== schedule.numPistons) {
      throw new Error("pistonCells length must equal schedule.numPistons");
    }
    this.schedP = schedule.numPistons;
    this.schedT = schedule.numSteps;

    this.cellsBuf?.destroy();
    this.ampsBuf?.destroy();
    this.cellsBuf = this.device.createBuffer({
      label: "m5.pistonCells",
      size: pistonCells.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    // Copy into fresh ArrayBuffer-backed views (writeBuffer wants ArrayBuffer,
    // not the ArrayBufferLike that TypedArray params are typed as).
    this.device.queue.writeBuffer(this.cellsBuf, 0, Uint32Array.from(pistonCells));

    this.ampsBuf = this.device.createBuffer({
      label: "m5.pistonAmps",
      size: schedule.a.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.ampsBuf, 0, Float32Array.from(schedule.a));

    this.scheduleLoaded = true;
  }

  get numSteps(): number {
    return this.schedT;
  }

  /** Zero the height ring (and injection texture) to replay from rest. */
  reset(): void {
    const encoder = this.device.createCommandEncoder({ label: "m5.reset" });
    for (const tex of [this.ring.prev, this.ring.curr, this.ring.next, this.injTex]) {
      const bind = this.device.createBindGroup({
        layout: this.clearPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: tex.createView() },
          { binding: 1, resource: { buffer: this.clearUniform } },
        ],
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.clearPipeline);
      pass.setBindGroup(0, bind);
      const g = dispatchCount(this.params.n, WG);
      pass.dispatchWorkgroups(g, g);
      pass.end();
    }
    this.device.queue.submit([encoder.finish()]);
    this.step = 0;
  }

  private writeParams(): void {
    const c = waveSpeed(this.params);
    const dv = new DataView(this.paramData);
    dv.setUint32(0, this.params.n, true);
    dv.setFloat32(4, this.params.dx, true);
    dv.setFloat32(8, c * c * this.dt * this.dt, true); // c²Δt²
    dv.setFloat32(12, 0.5 * this.params.gamma * this.dt, true); // ½γΔt
    const pk = this.queuedPoke;
    dv.setFloat32(16, pk ? pk.x : 0, true);
    dv.setFloat32(20, pk ? pk.y : 0, true);
    dv.setFloat32(24, pk ? pk.radius : 1, true);
    dv.setFloat32(28, pk ? pk.amplitude : 0, true);
    this.device.queue.writeBuffer(this.uniform, 0, this.paramData);
    this.queuedPoke = null;
  }

  /**
   * Encode one leapfrog step. If `pistonStep` is given and a schedule is loaded,
   * that step's piston amplitudes are injected (scatter → wave-step add).
   */
  encode(encoder: GPUCommandEncoder, pistonStep: number | null = null): void {
    this.writeParams();

    // Scatter this step's amplitudes into the injection texture (before the wave
    // step reads it). When not playing, injTex stays zero from the last reset.
    if (pistonStep !== null && this.scheduleLoaded && this.cellsBuf && this.ampsBuf) {
      const su = new ArrayBuffer(16);
      const sdv = new DataView(su);
      sdv.setUint32(0, this.schedP, true);
      sdv.setUint32(4, this.schedT, true);
      sdv.setUint32(8, Math.max(0, Math.min(this.schedT - 1, pistonStep)), true);
      sdv.setUint32(12, this.params.n, true);
      this.device.queue.writeBuffer(this.scatterUniform, 0, su);

      const sbind = this.device.createBindGroup({
        layout: this.scatterPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.injTex.createView() },
          { binding: 1, resource: { buffer: this.cellsBuf } },
          { binding: 2, resource: { buffer: this.ampsBuf } },
          { binding: 3, resource: { buffer: this.scatterUniform } },
        ],
      });
      const spass = encoder.beginComputePass({ label: "m5.scatter" });
      spass.setPipeline(this.scatterPipeline);
      spass.setBindGroup(0, sbind);
      spass.dispatchWorkgroups(dispatchCount(this.schedP, WG_LINE));
      spass.end();
    }

    const bindGroup = this.device.createBindGroup({
      label: "m5.bind",
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.ring.curr.createView() },
        { binding: 1, resource: this.ring.prev.createView() },
        { binding: 2, resource: this.ring.next.createView() },
        { binding: 3, resource: { buffer: this.uniform } },
        { binding: 4, resource: this.injTex.createView() },
      ],
    });

    const pass = encoder.beginComputePass({ label: "m5.wave_step" });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    const groups = dispatchCount(this.params.n, WG);
    pass.dispatchWorkgroups(groups, groups);
    pass.end();

    this.ring.swap(); // h^{n+1} becomes the new h^n
    this.step++;
  }

  /** Current surface, for M6. heightTex is the freshly-computed h^n. */
  get state(): FluidState {
    return {
      width: this.params.n,
      height: this.params.n,
      heightTex: this.ring.curr,
      normalTex: this.normalTex,
      cellSize: this.params.dx,
    };
  }

  destroy(): void {
    this.ring.destroy();
    this.normalTex.destroy();
    this.uniform.destroy();
    this.injTex.destroy();
    this.scatterUniform.destroy();
    this.clearUniform.destroy();
    this.cellsBuf?.destroy();
    this.ampsBuf?.destroy();
  }
}
