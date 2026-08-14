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
import { ActuationMapper, type PinvMeta } from "./modules/m4_actuation/index.ts";
import { Stage3D } from "./modules/m8_stage3d/index.ts";
import { CpuWaterPlayer } from "./modules/m8_stage3d/water_player.ts";
import { CausticPainter } from "./modules/m8_stage3d/floor_caustic.ts";
import type { PistonSchedule } from "./contracts/index.ts";
import type { WaveParams } from "./physics.ts";
// 32²/24-piston actuation asset: geometry + golden sample in JSON, the ~3.9 MB
// pseudoinverse fetched as a binary (too big to inline in the bundle).
import pinvMeta from "./modules/m4_actuation/__fixtures__/pinv_medium.json";
import pinvBinUrl from "./modules/m4_actuation/__fixtures__/pinv_medium.bin?url";

const errorEl = document.getElementById("error") as HTMLDivElement;
const hintEl = document.getElementById("hint") as HTMLDivElement;

function showError(err: unknown): void {
  const msg = err instanceof Error ? `${err.message}\n\n${err.stack ?? ""}` : String(err);
  errorEl.textContent = msg;
  console.error(err);
}

const FOCAL_D = 0.15;
const N_REL = 1.333;

/** Peak |h| of a heightmap. */
function peakAbs(hT: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < hT.length; i++) m = Math.max(m, Math.abs(hT[i]));
  return m;
}

async function boot(): Promise<void> {
  const meta = pinvMeta as unknown as PinvMeta;
  const g = meta.geometry;
  const pistonCells = Uint32Array.from(g.pistonCells);

  // Fetch the binary pseudoinverse and build the actuation mapper from it.
  const pinvData = new Float32Array(await (await fetch(pinvBinUrl)).arrayBuffer());
  const mapper = ActuationMapper.fromBinary(g, pinvData);

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
  // M-C: wavemaker paddles at the perimeter piston cells, driven in lockstep.
  stage.buildPistons(pistonCells, g.n);
  // M-D: the caustic on the 3D floor is recomputed on the CPU from the SAME water
  // field (a WebGPU canvas can't be sampled as a WebGL texture), painted to a 2D
  // canvas the floor displays.
  const floorCaustic = new CausticPainter({ d: FOCAL_D, nRel: N_REL, dx: g.dx });
  stage.setFloorCaustic(floorCaustic.canvas);

  const heightBuf = new Float32Array(g.n * g.n);
  const pistonBuf = new Float32Array(g.pistonCount);
  // Smooth, frame-rate-independent playback: feed the real frame dt so the water
  // and paddles interpolate between physics steps (no staircase jitter). The floor
  // caustic is the expensive per-frame cost, so repaint it every other frame.
  let frameCount = 0;
  stage.onFrame = (dt) => {
    player.tick(dt);
    const field = player.renderHeight(heightBuf);
    stage.displaceWater(field, player.n);
    stage.setPistonOffsets(player.renderPistons(pistonBuf));
    if (frameCount++ % 2 === 0) floorCaustic.paint(field, player.n);
  };
  // NOTE: the 3D stage is NOT self-started; a single app loop (below) drives both
  // the 3D tank and the 2D preview off ONE dt so they never drift apart.

  const gpuCanvas = document.getElementById("gpu-canvas") as HTMLCanvasElement;
  const gpu = await initGpu(gpuCanvas);

  // Caustic view runs at the asset geometry, in Pulse mode.
  const orch = new Orchestrator(gpu, {
    wave: { n: g.n, dx: g.dx, depth: g.depth, gamma: g.gamma, cflSafety: 0.9 },
    // renderRes 256 (up from 128) → a smoother, higher-quality preview caustic;
    // per-splat energy auto-scales so brightness is unchanged. Exposure raised so
    // the preview reads about as bright as the 3D floor caustic.
    render: { focalDistance: FOCAL_D, nRel: N_REL, cellEnergy: 0.5, exposure: 2.6, renderRes: 256 },
  });

  // Inverse solver (geometry-fixed → build once). `mapper` was built above.
  const solver = new InverseCausticSolver(g.n);

  const runTarget = (hT: Float32Array | number[], label: string): void => {
    // Use the NATURAL solved surface — it is physically designed to focus the
    // target at the tank floor. Do NOT rescale it: amplifying a thin feature's
    // small relief over-drives the refraction so the rays over-deflect and smear
    // the caustic into multiple spread lines instead of one clean one.
    const schedule: PistonSchedule = mapper.solve(hT);
    orch.startPulse(schedule, pistonCells); // 2D caustic preview
    player.load(schedule, pistonCells, waveParams); // 3D water surface

    // The caustic uses the natural amplitude, but the 3D water/paddle DISPLAY is
    // scaled per-solve so even a thin drawing's tiny ridge is visible in the tank
    // (this is cosmetic exaggeration of the mesh only — it does not touch the
    // physics or the caustic).
    const peak = peakAbs(hT) || 1;
    stage.waterVerticalScale = (0.16 * stage.cfg.tankDepth) / peak;
    const pAmp = player.maxAbsAmplitude();
    stage.pistonTravelScale = pAmp > 0 ? (0.14 * stage.cfg.tankSize) / pAmp : 0.15;
    hintEl.textContent = `${label} — the caustic pulses into the target. Draw again and Solve to change it.`;
  };

  const solveFromSketch = (intensity: Float32Array): void => {
    // Higher contrast than the paraxial default allowed — the Monge-Ampère solve
    // stays accurate here (see m3.test), giving crisper, more complex caustics.
    const density = preprocessDensity(intensity, g.n, {
      blurSigma: 0.9, // keep fine strokes (e.g. a thin line) narrow
      // Low ambient floor so the caustic has a DARK background with the drawing
      // bright on top (like the zero-mean demo bump), instead of a fully-lit floor
      // the drawing barely rises above. Measured: this also improves how faithfully
      // the caustic matches the drawing. Must stay > 0 (solver needs I > 0).
      ambient: 0.35,
      gain: 1.5,
    });
    const { target } = solver.solveMA(Float64Array.from(density.I), g.dx, FOCAL_D, N_REL);
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
      runTarget(meta.sample!.hT, "Demo bump");
    } catch (err) {
      showError(err);
    }
  });

  // Show the demo target immediately so the caustic isn't blank on load.
  runTarget(meta.sample!.hT, "Demo bump");

  // Single clock for BOTH views: identical dt in → identical pulse progression,
  // so the 3D floor and the 2D preview stay locked together even if a frame is
  // heavy (a heavy frame slows both equally, never one relative to the other).
  let lastT = performance.now();
  const loop = (now: number) => {
    const dt = Math.min(0.05, (now - lastT) / 1000);
    lastT = now;
    try {
      stage.frame(dt); // 3D tank (drives the CPU water player via onFrame)
      orch.frame(dt); // 2D caustic preview
    } catch (err) {
      showError(err);
      return;
    }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

boot().catch(showError);
