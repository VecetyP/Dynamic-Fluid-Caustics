/**
 * M5 · Fluid Simulation Engine.
 *
 * Advances the damped wave field one step per frame (eq 4.7) on a ping-pong of
 * r32float textures, and exposes the current surface as the height half of a
 * FluidState. Normal derivation lives in M6 (it consumes this height texture).
 */

import type { FluidState } from "../../contracts/index.ts";
import { createHeightPingPong, type PingPong } from "../../gpu/pingpong.ts";
import { createComputePipeline, createStorageTexture, dispatchCount } from "../../gpu/helpers.ts";
import { assertCflStable, chooseDt, waveSpeed, type WaveParams } from "../../physics.ts";
import waveStepWgsl from "./wave_step.wgsl?raw";

const WG = 16; // workgroup_size in the shader
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
    this.normalTex = createStorageTexture(
      device,
      params.n,
      "rgba16float",
      "m5.normals"
    );
  }

  /** Queue a Gaussian displacement to be injected on the next step. */
  poke(p: PendingPoke): void {
    this.queuedPoke = p;
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

  /** Encode one leapfrog step. Advances the ping-pong ring. */
  encode(encoder: GPUCommandEncoder): void {
    this.writeParams();

    const bindGroup = this.device.createBindGroup({
      label: "m5.bind",
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.ring.curr.createView() },
        { binding: 1, resource: this.ring.prev.createView() },
        { binding: 2, resource: this.ring.next.createView() },
        { binding: 3, resource: { buffer: this.uniform } },
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
  }
}
