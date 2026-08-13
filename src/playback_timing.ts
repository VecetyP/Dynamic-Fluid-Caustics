/**
 * Shared playback pacing for the pulse loop, in WALL-CLOCK seconds (at speed 1).
 *
 * Both the 3D water player (`m8_stage3d/water_player.ts`) and the 2D caustic
 * preview orchestrator (`m7_orchestrator`) key off these so the two views build,
 * focus, hold, and re-pulse in step. Scaled by the speed slider.
 */

/** Seconds of wall-clock per physics step during build-up. */
export const BASE_STEP_SECONDS = 0.18; // ~3.6 s to build 20 steps at speed 1
/** Seconds to hold the focused (focal) surface before re-pulsing. */
export const BASE_HOLD_SECONDS = 2.0;
