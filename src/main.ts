/**
 * Entry point.
 *   - Default: interactive ripple → caustic (Phase 1). Click/drag to poke.
 *   - Press "P": Phase-3 pulse demo. Solves a piston schedule for a target
 *     surface (M4) and replays it (M5), so the caustic focuses into the target
 *     each pulse. Runs at the M4 asset's geometry. Press "I" to return.
 */

import { initGpu, type GpuContext } from "./gpu/device.ts";
import { Orchestrator } from "./modules/m7_orchestrator/index.ts";
import { ActuationMapper, type PinvAsset } from "./modules/m4_actuation/index.ts";
import pinvAsset from "./modules/m4_actuation/__fixtures__/pinv_small.json";

const errorEl = document.getElementById("error") as HTMLDivElement;
const hintEl = document.getElementById("hint") as HTMLDivElement;

function showError(err: unknown): void {
  const msg = err instanceof Error ? `${err.message}\n\n${err.stack ?? ""}` : String(err);
  errorEl.textContent = msg;
  console.error(err);
}

function makeInteractive(gpu: GpuContext): Orchestrator {
  const orch = new Orchestrator(gpu);
  orch.pokeNormalised(0.5, 0.5, 0.8); // seed a ripple
  hintEl.textContent =
    "Interactive: click/drag to poke the surface. Press P for the piston caustic demo.";
  return orch;
}

function makePulseDemo(gpu: GpuContext): Orchestrator {
  const asset = pinvAsset as PinvAsset;
  const g = asset.geometry;
  // Run M5 at the asset's exact geometry so the schedule reproduces on the GPU.
  const orch = new Orchestrator(gpu, {
    wave: { n: g.n, dx: g.dx, depth: g.depth, gamma: g.gamma, cflSafety: 0.9 },
    render: { focalDistance: 0.15, nRel: 1.333, cellEnergy: 0.5, exposure: 1.2 },
  });

  // Full inverse step: target surface → piston schedule (M4).
  const schedule = new ActuationMapper(asset).solve(asset.sample!.hT);
  const pistonCells = Uint32Array.from(g.pistonCells);
  orch.startPulse(schedule, pistonCells);

  hintEl.textContent =
    `Pulse demo (${g.n}×${g.n}, ${g.pistonCount} pistons): the caustic focuses into the ` +
    `target each cycle. Press I for interactive ripple.`;
  return orch;
}

async function boot(): Promise<void> {
  const canvas = document.getElementById("gpu-canvas") as HTMLCanvasElement;
  const gpu = await initGpu(canvas);

  let active: Orchestrator = makeInteractive(gpu);

  // Pointer → poke (interactive mode only).
  let dragging = false;
  const pokeAt = (e: PointerEvent) => {
    if (active.state !== "INTERACTIVE") return;
    const r = canvas.getBoundingClientRect();
    const u = (e.clientX - r.left) / r.width;
    const v = (e.clientY - r.top) / r.height;
    if (u >= 0 && u <= 1 && v >= 0 && v <= 1) active.pokeNormalised(u, v);
  };
  canvas.addEventListener("pointerdown", (e) => {
    dragging = true;
    pokeAt(e);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (dragging) pokeAt(e);
  });
  window.addEventListener("pointerup", () => (dragging = false));

  // Mode toggle.
  window.addEventListener("keydown", (e) => {
    const key = e.key.toLowerCase();
    if (key === "p" && active.state !== "PULSE") {
      active.destroy();
      active = makePulseDemo(gpu);
    } else if (key === "i" && active.state !== "INTERACTIVE") {
      active.destroy();
      active = makeInteractive(gpu);
    }
  });

  const loop = () => {
    try {
      active.frame();
    } catch (err) {
      showError(err);
      return; // stop the loop on error
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

boot().catch(showError);
