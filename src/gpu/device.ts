/** WebGPU device acquisition + canvas context config. */

export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  context: GPUCanvasContext;
  /** Preferred swap-chain format for the canvas. */
  format: GPUTextureFormat;
}

export async function initGpu(canvas: HTMLCanvasElement): Promise<GpuContext> {
  if (!("gpu" in navigator)) {
    throw new Error(
      "WebGPU is not available in this browser. Use Chrome/Edge 113+, or " +
        "enable the flag in Firefox/Safari Technology Preview."
    );
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) throw new Error("No WebGPU adapter found.");

  const device = await adapter.requestDevice();
  device.lost.then((info) => {
    // Surfaced to the page by main.ts's error handler on next frame.
    console.error("WebGPU device lost:", info.message, info.reason);
  });

  const context = canvas.getContext("webgpu");
  if (!context) throw new Error("Failed to get a 'webgpu' canvas context.");

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "premultiplied" });

  return { adapter, device, canvas, context, format };
}
