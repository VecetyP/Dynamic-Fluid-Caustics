/**
 * Entry point — wires the canvas to the Phase-1 pipeline (M5 sim + M6 render via
 * M7). Click/drag pokes the surface; the caustic accumulates on the floor plane.
 */

import { initGpu } from "./gpu/device.ts";
import { Orchestrator } from "./modules/m7_orchestrator/index.ts";

const errorEl = document.getElementById("error") as HTMLDivElement;

function showError(err: unknown): void {
  const msg = err instanceof Error ? `${err.message}\n\n${err.stack ?? ""}` : String(err);
  errorEl.textContent = msg;
  console.error(err);
}

async function boot(): Promise<void> {
  const canvas = document.getElementById("gpu-canvas") as HTMLCanvasElement;
  const gpu = await initGpu(canvas);
  const orch = new Orchestrator(gpu);

  // Pointer → poke (normalised canvas coords).
  let dragging = false;
  const pokeAt = (e: PointerEvent) => {
    const r = canvas.getBoundingClientRect();
    const u = (e.clientX - r.left) / r.width;
    const v = (e.clientY - r.top) / r.height;
    if (u >= 0 && u <= 1 && v >= 0 && v <= 1) orch.pokeNormalised(u, v);
  };
  canvas.addEventListener("pointerdown", (e) => {
    dragging = true;
    pokeAt(e);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (dragging) pokeAt(e);
  });
  window.addEventListener("pointerup", () => (dragging = false));

  // Seed one ripple so there's something on screen immediately.
  orch.pokeNormalised(0.5, 0.5, 0.8);

  const loop = () => {
    try {
      orch.frame();
    } catch (err) {
      showError(err);
      return; // stop the loop on error
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

boot().catch(showError);
