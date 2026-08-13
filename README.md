# Dynamic Fluid Caustics

Interactive wave-control & inverse-rendering caustic simulator — WebGPU + TypeScript.
See `Dynamic_Fluid_Caustics_Technical_Specification.pdf` for the full design and
`Project_Kickoff.md` for the build plan.

## Status: Phase 1 (Engine Foundation)

Working per-frame path — poke the water surface, watch a damped ripple propagate,
reflect off the walls, and cast a moving caustic on the floor plane.

- **M5** damped shallow-water wave sim (leapfrog FDM, eq 4.7) on ping-pong r32float textures
- **M6** GPU render: normal derivation → refraction splat (additive blend, spec path B) → tone map
- **M7** minimal orchestrator + render loop

The inverse-caustic solver (M3), actuation mapper (M4), density preprocessor (M2),
and drawing canvas (M1) are Phases 2–4 — not yet implemented.

## Run

```bash
npm install
npm run dev      # opens http://localhost:5173
```

Requires a WebGPU browser (Chrome/Edge 113+, or Firefox/Safari with the flag).

## Verify

```bash
npm run typecheck   # tsc, no emit
npm run test        # vitest — CFL + leapfrog reference (golden oracle)
```

## Layout

```
src/
  contracts/    §3.2 typed payloads (module boundaries)
  physics.ts    shared wave math + CFL (framework-free, unit-tested)
  gpu/          device init, ping-pong textures, pipeline helpers
  modules/
    m5_fluid/   wave sim: wave_step.wgsl + CPU reference + tests
    m6_render/  normals.wgsl, caustic.wgsl, tone.wgsl
    m7_orchestrator/  state machine + frame loop
  main.ts       canvas wiring, pointer input, error surface
```

## Key design notes

- **Path B, not path A** for the caustic. WGSL has no float/texture atomics, so
  the spec's `imageAtomicAdd` scatter doesn't port; additive-blend point splatting
  is the race-free WebGPU equivalent.
- **CFL asserted at init** (`assertCflStable`). Explicit leapfrog diverges if the
  timestep violates eq 4.8, so the sim refuses to start rather than blow up.
- **Reflective walls** via clamped neighbour loads (∂h/∂n = 0), branch-free.
- Grid defaults to **128²**; raise `n` in `physics.ts` once the pipeline is proven.
