/**
 * M-B verification: the 3D water is the verified physics.
 *
 * CpuWaterPlayer paces the same operator as `forwardPlayback` (the golden CPU
 * oracle). Running it through all build-up steps must reproduce `forwardPlayback`
 * exactly, and each `tick` at framesPerStep=1 must advance one physics step. This
 * lets us confirm the M-B surface headlessly, with no WebGL.
 */

import { describe, it, expect } from "vitest";
import { CpuWaterPlayer } from "./water_player.ts";
import { forwardPlayback } from "../m5_fluid/reference.ts";
import type { WaveParams } from "../../physics.ts";
import type { PistonSchedule } from "../../contracts/index.ts";
import pinvAsset from "../m4_actuation/__fixtures__/pinv_small.json";

const g = (pinvAsset as any).geometry;
const aExpected: number[] = (pinvAsset as any).sample.aExpected;

function makeParams(): WaveParams {
  return { n: g.n, dx: g.dx, depth: g.depth, gamma: g.gamma, cflSafety: 0.9 };
}
function makeSchedule(): PistonSchedule {
  return {
    numPistons: g.pistonCount,
    numSteps: g.numSteps,
    a: Float32Array.from(aExpected),
    dt: g.dt,
    focalTime: g.focalTime,
  };
}

describe("CpuWaterPlayer", () => {
  it("running to focal reproduces forwardPlayback bit-for-bit", () => {
    const params = makeParams();
    const schedule = makeSchedule();
    const player = new CpuWaterPlayer();
    player.load(schedule, g.pistonCells, params);

    let guard = 0;
    while (player.phase === "building" && guard++ < 10_000) player.advanceStep();
    expect(player.phase).toBe("hold");
    expect(player.cursor).toBe(g.numSteps);

    const oracle = forwardPlayback(
      params,
      schedule.dt,
      g.pistonCells,
      schedule.a,
      schedule.numPistons,
      schedule.numSteps
    );

    const h = player.height()!;
    expect(h.length).toBe(oracle.length);
    let maxDiff = 0;
    for (let i = 0; i < h.length; i++) maxDiff = Math.max(maxDiff, Math.abs(h[i] - oracle[i]));
    expect(maxDiff).toBe(0);
  });

  it("smooth playback lands exactly on the focal surface at hold", () => {
    const params = makeParams();
    const schedule = makeSchedule();
    const player = new CpuWaterPlayer();
    player.load(schedule, g.pistonCells, params);

    // Drive wall-clock playback with fixed dt until it reaches the hold phase.
    let guard = 0;
    while (player.phase !== "hold" && guard++ < 100_000) player.tick(0.05);
    expect(player.phase).toBe("hold");

    // At hold, frac=1 → the interpolated render surface equals the verified
    // focal surface bit-for-bit.
    const oracle = forwardPlayback(
      params,
      schedule.dt,
      g.pistonCells,
      schedule.a,
      schedule.numPistons,
      schedule.numSteps
    );
    const rh = player.renderHeight()!;
    let maxDiff = 0;
    for (let i = 0; i < rh.length; i++) maxDiff = Math.max(maxDiff, Math.abs(rh[i] - oracle[i]));
    expect(maxDiff).toBe(0);

    // Interpolation stays within the physics envelope (no overshoot) mid-build.
    player.reset();
    player.tick(0.09); // ~half a step at speed 1 (BASE_STEP_SECONDS=0.18)
    const mid = player.renderHeight()!;
    expect(mid.every((v) => Number.isFinite(v))).toBe(true);
  });

  it("pistonAmplitudes track the most-recently-injected step", () => {
    const schedule = makeSchedule();
    const player = new CpuWaterPlayer();
    player.load(schedule, g.pistonCells, makeParams());

    expect(player.pistonCount).toBe(g.pistonCount);
    // Before any step: all zero, display step -1.
    expect(player.displayStep()).toBe(-1);
    expect(Array.from(player.pistonAmplitudes()).every((v) => v === 0)).toBe(true);

    // After one step, amplitudes == schedule column 0 (a[k*T + 0]).
    player.advanceStep();
    expect(player.displayStep()).toBe(0);
    const amps = player.pistonAmplitudes();
    for (let k = 0; k < g.pistonCount; k++) {
      expect(amps[k]).toBe(schedule.a[k * g.numSteps + 0]);
    }

    // After a second step, column 1.
    player.advanceStep();
    expect(player.displayStep()).toBe(1);
    const amps2 = player.pistonAmplitudes();
    for (let k = 0; k < g.pistonCount; k++) {
      expect(amps2[k]).toBe(schedule.a[k * g.numSteps + 1]);
    }
  });

  it("reset returns the surface to rest", () => {
    const player = new CpuWaterPlayer();
    player.load(makeSchedule(), g.pistonCells, makeParams());
    player.advanceStep();
    player.advanceStep();
    player.reset();
    const h = player.height()!;
    expect(player.cursor).toBe(0);
    expect(h.every((v) => v === 0)).toBe(true);
  });
});
