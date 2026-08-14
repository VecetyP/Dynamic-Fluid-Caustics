/**
 * Shared playback pacing for the pulse loop, in WALL-CLOCK seconds (at speed 1).
 *
 * Both the 3D water player (`m8_stage3d/water_player.ts`) and the 2D caustic
 * preview orchestrator (`m7_orchestrator`) key off these so the two views build,
 * focus, hold, and re-pulse in step. Scaled by the speed slider.
 *
 * The build-up is timed as a TOTAL duration and split across however many steps
 * the schedule has (`BUILD_SECONDS / numSteps` per step), so changing the step
 * count (e.g. a finer asset with more steps) doesn't change how long the pulse
 * takes to watch.
 */

/** Total wall-clock seconds to play the whole build-up (all steps), at speed 1. */
export const BUILD_SECONDS = 3.5;
/** Seconds to hold the focused (focal) surface before re-pulsing. */
export const HOLD_SECONDS = 2.0;

/** Per-step duration for a schedule of `numSteps` steps at playback `speed`. */
export function stepSeconds(numSteps: number, speed: number): number {
  const n = Math.max(1, numSteps);
  return BUILD_SECONDS / (Math.max(0.1, speed) * n);
}

/** Hold duration at playback `speed`. */
export function holdSeconds(speed: number): number {
  return HOLD_SECONDS / Math.max(0.1, speed);
}
