/**
 * M6 · GPU Render Pipeline.
 *
 * Per frame: (1) derive surface normals from M5's height texture; (2) splat one
 * refracted ray per RENDER cell onto a floating-point floor accumulator via
 * additive blending (path B); (3) tone-map the accumulator to the swap-chain.
 *
 * The render grid (`renderRes`) is decoupled from the sim grid: normals are
 * bilinearly sampled, so a coarse sim yields a high-resolution caustic.
 */

import type { FluidState } from "../../contracts/index.ts";
import { createComputePipeline, dispatchCount } from "../../gpu/helpers.ts";
import normalsWgsl from "./normals.wgsl?raw";
import causticWgsl from "./caustic.wgsl?raw";
import toneWgsl from "./tone.wgsl?raw";

const WG = 16;

export interface RenderOptions {
  focalDistance: number; // surface → floor (m)
  nRel: number; // relative refractive index n2/n1
  cellEnergy: number; // energy per splat (auto-scaled for renderRes)
  exposure: number; // tone-map exposure
  renderRes?: number; // caustic render grid dim (default 128)
}

export class CausticRenderer {
  private readonly device: GPUDevice;
  private readonly n: number;
  private readonly res: number;

  private readonly normalPipe: GPUComputePipeline;
  private readonly causticPipe: GPURenderPipeline;
  private readonly tonePipe: GPURenderPipeline;

  private readonly nParams: GPUBuffer;
  private readonly cParams: GPUBuffer;
  private readonly tParams: GPUBuffer;

  private readonly accum: GPUTexture;
  private readonly sampler: GPUSampler;

  private causticBind!: GPUBindGroup;
  private toneBind!: GPUBindGroup;

  constructor(device: GPUDevice, n: number, dx: number, format: GPUTextureFormat, opts: RenderOptions) {
    this.device = device;
    this.n = n;
    this.res = opts.renderRes ?? 128;

    this.normalPipe = createComputePipeline(device, normalsWgsl, "m6.normals");

    // Floating-point floor accumulator at RENDER resolution (path B target).
    this.accum = device.createTexture({
      label: "m6.floor_accum",
      size: { width: this.res, height: this.res },
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.causticPipe = device.createRenderPipeline({
      label: "m6.caustic",
      layout: "auto",
      vertex: { module: device.createShaderModule({ code: causticWgsl }), entryPoint: "vs" },
      fragment: {
        module: device.createShaderModule({ code: causticWgsl }),
        entryPoint: "fs",
        targets: [
          {
            format: "rgba16float",
            blend: {
              color: { srcFactor: "one", dstFactor: "one", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "point-list" },
    });

    this.tonePipe = device.createRenderPipeline({
      label: "m6.tone",
      layout: "auto",
      vertex: { module: device.createShaderModule({ code: toneWgsl }), entryPoint: "vs" },
      fragment: {
        module: device.createShaderModule({ code: toneWgsl }),
        entryPoint: "fs",
        targets: [{ format }],
      },
      primitive: { topology: "triangle-list" },
    });

    this.nParams = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.cParams = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.tParams = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    // NParams { n, dx }
    const nb = new ArrayBuffer(16);
    const ndv = new DataView(nb);
    ndv.setUint32(0, n, true);
    ndv.setFloat32(4, dx, true);
    device.queue.writeBuffer(this.nParams, 0, nb);

    // CParams { n, res, dx, d, nRel, cellEnergy, pad, pad }.
    // Scale per-ray energy by (n/res)² so total deposited energy — hence overall
    // brightness — is independent of the render resolution.
    const energy = opts.cellEnergy * (n * n) / (this.res * this.res);
    const cb = new ArrayBuffer(32);
    const cdv = new DataView(cb);
    cdv.setUint32(0, n, true);
    cdv.setUint32(4, this.res, true);
    cdv.setFloat32(8, dx, true);
    cdv.setFloat32(12, opts.focalDistance, true);
    cdv.setFloat32(16, opts.nRel, true);
    cdv.setFloat32(20, energy, true);
    device.queue.writeBuffer(this.cParams, 0, cb);

    this.setExposure(opts.exposure);
    this.sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });
  }

  setExposure(exposure: number): void {
    const tb = new ArrayBuffer(16);
    new DataView(tb).setFloat32(0, exposure, true);
    this.device.queue.writeBuffer(this.tParams, 0, tb);
  }

  private ensureBinds(fluid: FluidState): void {
    if (!this.causticBind) {
      this.causticBind = this.device.createBindGroup({
        layout: this.causticPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: fluid.normalTex.createView() },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: { buffer: this.cParams } },
        ],
      });
    }
    if (!this.toneBind) {
      this.toneBind = this.device.createBindGroup({
        layout: this.tonePipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.accum.createView() },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: { buffer: this.tParams } },
        ],
      });
    }
  }

  encode(encoder: GPUCommandEncoder, fluid: FluidState, swapView: GPUTextureView): void {
    this.ensureBinds(fluid);

    // (1) Normal pass — height (rotates each frame) → normalTex.
    const normalBind = this.device.createBindGroup({
      layout: this.normalPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: fluid.heightTex.createView() },
        { binding: 1, resource: fluid.normalTex.createView() },
        { binding: 2, resource: { buffer: this.nParams } },
      ],
    });
    const np = encoder.beginComputePass({ label: "m6.normals" });
    np.setPipeline(this.normalPipe);
    np.setBindGroup(0, normalBind);
    const g = dispatchCount(this.n, WG);
    np.dispatchWorkgroups(g, g);
    np.end();

    // (2) Caustic splat — clear accumulator, then additive-blend one point per
    // render cell (renderRes²).
    const cp = encoder.beginRenderPass({
      label: "m6.caustic",
      colorAttachments: [
        {
          view: this.accum.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    cp.setPipeline(this.causticPipe);
    cp.setBindGroup(0, this.causticBind);
    cp.draw(this.res * this.res);
    cp.end();

    // (3) Tone pass → swap-chain.
    const tp = encoder.beginRenderPass({
      label: "m6.tone",
      colorAttachments: [
        { view: swapView, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
      ],
    });
    tp.setPipeline(this.tonePipe);
    tp.setBindGroup(0, this.toneBind);
    tp.draw(3);
    tp.end();
  }

  destroy(): void {
    this.accum.destroy();
    this.nParams.destroy();
    this.cParams.destroy();
    this.tParams.destroy();
  }
}
