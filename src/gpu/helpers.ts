/** Thin WebGPU pipeline / texture helpers to keep module code readable. */

export function createComputePipeline(
  device: GPUDevice,
  code: string,
  label: string,
  entryPoint = "main"
): GPUComputePipeline {
  return device.createComputePipeline({
    label,
    layout: "auto",
    compute: { module: device.createShaderModule({ label: `${label}.wgsl`, code }), entryPoint },
  });
}

/** Divide-and-round-up for compute dispatch grid sizing. */
export function dispatchCount(total: number, workgroup: number): number {
  return Math.ceil(total / workgroup);
}

export function createStorageTexture(
  device: GPUDevice,
  size: number,
  format: GPUTextureFormat,
  label: string,
  extraUsage: GPUTextureUsageFlags = 0
): GPUTexture {
  return device.createTexture({
    label,
    size: { width: size, height: size },
    format,
    usage:
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.TEXTURE_BINDING |
      extraUsage,
  });
}
