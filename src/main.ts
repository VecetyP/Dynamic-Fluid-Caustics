/**
 * Entry point — the full sketch → caustic loop (Phase 4).
 *
 *   draw (M1) → density (M2) → surface (M3) → piston schedule (M4)
 *             → wave replay (M5) → caustic (M6), pulsed by M7.
 *
 * Everything runs at the shipped M⁺ asset's geometry so the precomputed inverse
 * applies. Draw a shape, press Solve, and the caustic pulses into it.
 */

import { initGpu } from "./gpu/device.ts";
import { Orchestrator } from "./modules/m7_orchestrator/index.ts";
import { DrawingCanvas } from "./modules/m1_canvas/index.ts";
import { preprocessDensity } from "./modules/m2_density/index.ts";
import { InverseCausticSolver } from "./modules/m3_inverse/index.ts";
import { ActuationMapper, type PinvAsset } from "./modules/m4_actuation/index.ts";
import { Stage3D } from "./modules/m8_stage3d/index.ts";
import { CpuWaterPlayer } from "./modules/m8_stage3d/water_player.ts";
import type { PistonSchedule } from "./contracts/index.ts";
import type { WaveParams } from "./physics.ts";
import pinvAsset from "./modules/m4_actuation/__fixtures__/pinv_small.json";

const errorEl = document.getElementById("error") as HTMLDivElement;
const hintEl = document.getElementById("hint") as HTMLDivElement;

function showError(err: unknown): void {
  const msg = err instanceof Error ? `${err.message}\n\n${err.stack ?? ""}` : String(err);
  errorEl.textContent = msg;
  console.error(err);
}

const FOCAL_D = 0.15;
const N_REL = 1.333;

/** Scale a heightmap to a target peak amplitude. The inverse solve fixes the
 *  surface SHAPE; its absolute relief depends on the target's contrast, which is
 *  tiny for a gentle sketch → a near-flat surface that barely refracts (uniform
 *  caustic). Normalising every surface to a common, strong relief makes any
 *  sketch bend light as decisively as the demo bump, so the caustic shows
 *  structure. M4 is linear, so this just scales the piston schedule. */
function normalizePeak(hT: ArrayLike<number>, peak: number): Float32Array {
  let maxAbs = 0;
  for (let i = 0; i < hT.length; i++) maxAbs = Math.max(maxAbs, Math.abs(hT[i]));
  const s = maxAbs > 0 ? peak / maxAbs : 1;
  const out = new Float32Array(hT.length);
  for (let i = 0; i < hT.length; i++) out[i] = hT[i] * s;
  return out;
}

async function boot(): Promise<void> {
  const asset = pinvAsset as unknown as PinvAsset;
  const g = asset.geometry;
  const pistonCells = Uint32Array.from(g.pistonCells);

  // 3D stage (M-A) — a static tank you can orbit. Independent of the WebGPU
  // pipeline below; later milestones (M-B…M-D) drive its water/pistons/floor
  // from the same physics. Started first so the viewport is live immediately.
  const stageCanvas = document.getElementById("stage3d") as HTMLCanvasElement;
  const stage = new Stage3D(stageCanvas, { gridN: g.n });

  // M-B: the CPU wave player drives the 3D water surface (reuses the verified
  // reference sim; no WebGPU↔WebGL bridge). It replays the same schedule the 2D
  // caustic uses, so both build up and hold in step.
  const waveParams: WaveParams = {
    n: g.n,
    dx: g.dx,
    depth: g.depth,
    gamma: g.gamma,
    cflSafety: 0.9,
  };
  const player = new CpuWaterPlayer();
  stage.onFrame = () => {
    player.tick();
    stage.displaceWater(player.height(), player.n);
  };
  stage.start();

  const gpuCanvas = document.getElementById("gpu-canvas") as HTMLCanvasElement;
  const gpu = await initGpu(gpuCanvas);

  // Caustic view runs at the asset geometry, in Pulse mode.
  const orch = new Orchestrator(gpu, {
    wave: { n: g.n, dx: g.dx, depth: g.depth, gamma: g.gamma, cflSafety: 0.9 },
    render: { focalDistance: FOCAL_D, nRel: N_REL, cellEnergy: 0.5, exposure: 1.2 },
  });

  // Inverse pipeline pieces (geometry-fixed → build once).
  const solver = new InverseCausticSolver(g.n);
  const mapper = new ActuationMapper(asset);

  // Common relief scale: the demo bump's own peak (it produces a good caustic).
  let demoPeak = 0;
  for (const v of asset.sample!.hT) demoPeak = Math.max(demoPeak, Math.abs(v));

  // Vertical exaggeration so the focal surface reaches a visible fraction of the
  // tank depth (the raw field is O(1) after normalizePeak).
  stage.waterVerticalScale = (0.18 * stage.cfg.tankDepth) / (demoPeak || 1);

  const runTarget = (hT: Float32Array | number[], label: string): void => {
    const schedule: PistonSchedule = mapper.solve(normalizePeak(hT, demoPeak));
    orch.startPulse(schedule, pistonCells); // 2D caustic preview
    player.load(schedule, pistonCells, waveParams); // 3D water surface
    hintEl.textContent = `${label} — the caustic pulses into the target. Draw again and Solve to change it.`;
  };

  const solveFromSketch = (intensity: Float32Array): void => {
    const density = preprocessDensity(intensity, g.n);
    const { target } = solver.solve(Float64Array.from(density.I), g.dx, FOCAL_D, N_REL);
    runTarget(target.hT, "Solved your sketch");
  };

  // Drawing pad (M1).
  const drawCanvas = document.getElementById("draw-canvas") as HTMLCanvasElement;
  const drawing = new DrawingCanvas(drawCanvas, 18);

  const speedEl = document.getElementById("speed") as HTMLInputElement;
  speedEl.addEventListener("input", () => {
    const s = parseFloat(speedEl.value);
    orch.setSpeed(s);
    player.setSpeed(s);
  });

  document.getElementById("solve-btn")!.addEventListener("click", () => {
    try {
      if (!drawing.hasContent()) {
        hintEl.textContent = "Draw something first, then Solve.";
        return;
      }
      solveFromSketch(drawing.sampleIntensity(g.n));
    } catch (err) {
      showError(err);
    }
  });
  document.getElementById("clear-btn")!.addEventListener("click", () => drawing.clear());
  document.getElementById("demo-btn")!.addEventListener("click", () => {
    try {
      runTarget(asset.sample!.hT, "Demo bump");
    } catch (err) {
      showError(err);
    }
  });

  // Show the demo target immediately so the caustic isn't blank on load.
  runTarget(asset.sample!.hT, "Demo bump");

  const loop = () => {
    try {
      orch.frame();
    } catch (err) {
      showError(err);
      return;
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

boot().catch(showError);
