# Dynamic Fluid Caustics

Draw a shape, and watch water focus light into it.

This is an interactive **inverse-caustics** simulator. You sketch (or will soon
upload) a target image; the software solves for the transient water-surface shape
whose refracted caustic reconstructs that image on the tank floor, computes the
wall-piston motions that produce that surface, runs a forward wave simulation, and
renders the resulting caustic — both as a 2D view and inside an orbitable 3D tank.

Built with **TypeScript + WebGPU** (2D physics/caustic) and **Three.js / WebGL**
(the 3D tank), bundled with Vite and tested with Vitest.

## What it does

Press **Solve** on a sketch and the pipeline runs end to end:

1. **Draw** a target on the canvas.
2. **Inverse solve** for the water surface that would refract light into that target
   (a Poisson/Neumann solve on the paraxial caustic equations).
3. **Actuation** — turn that surface into a per-piston, time-reversed amplitude
   schedule via a precomputed regularised pseudoinverse of the wave-response matrix.
4. **Forward simulate** the shallow-water waves driven by those pistons.
5. **Render the caustic** — at focal time the wavefronts converge and the caustic
   focuses into your drawing, then the whole thing re-pulses.

The interactive 3D tank shows the same physics: glass walls, wavemaker paddles
around the rim that move on the computed schedule, a rippling water surface, and the
live caustic on the tank floor.

> **Want the math?** [`HOW_IT_WORKS.md`](./HOW_IT_WORKS.md) is a self-contained deep
> dive into every step — the wave model, the caustic optics, the Poisson inverse, the
> pseudoinverse actuation, and the rendering — for readers who want to know how it
> works under the hood.

## Run

```bash
npm install
npm run dev      # opens http://localhost:5173
```

Requires a **WebGPU-capable browser** (Chrome/Edge 113+, or Firefox/Safari with the
WebGPU flag enabled). The 3D tank uses WebGL and runs everywhere.

## Verify

```bash
npm run typecheck   # tsc, no emit
npm run test        # vitest — solver, actuation, and closed-loop oracle tests
npm run build       # type-check + production bundle
```

The numerics are tested against CPU "golden oracle" references (and NumPy prototypes
under `prototypes/`), so the full inverse→forward loop is verified headlessly — no GPU
required to run the test suite.

## How it's built

The pipeline is split into small, single-responsibility modules that communicate
only through typed payloads (`src/contracts`):

```
src/
  contracts/    typed payloads that define the module boundaries
  physics.ts    shared shallow-water math + CFL stability (framework-free, tested)
  playback_timing.ts  shared wall-clock pacing for the 2D and 3D views
  gpu/          WebGPU device init, ping-pong textures, pipeline helpers
  modules/
    m1_canvas/       drawing canvas → greyscale target
    m2_density/      blur / band-limit → strictly-positive density map
    m3_inverse/      inverse-caustic Poisson solver (DCT, Neumann)
    m4_actuation/    wave-basis pseudoinverse → piston schedule
    m5_fluid/        forward wave sim (WGSL) + CPU reference oracle
    m6_render/       caustic render: normals → refraction splat → tone map
    m7_orchestrator/ pulse/hold/loop state machine + frame loop
    m8_stage3d/      the interactive 3D tank (Three.js): water, paddles,
                     floor caustic, camera
prototypes/     NumPy references used to validate the solvers
```

## Key design notes

- **Additive-blend caustic splatting.** WGSL has no float/texture atomics, so the
  classic scatter-with-`imageAtomicAdd` doesn't port; the caustic is rendered by
  additive-blend point splatting instead (race-free on WebGPU).
- **CFL asserted at init.** An explicit leapfrog integrator diverges if the timestep
  violates the CFL bound, so the sim refuses to start rather than blow up.
- **Reflective walls** via clamped neighbour loads (zero-gradient / Neumann), branch-free.
- **One shared operator.** The actuation basis, the GPU sim, and the CPU reference all
  use the exact same "advance, then inject" rule, so a solved schedule replays back to
  the target to machine precision — the property the closed-loop test checks.
- **The 3D tank runs the sim on the CPU.** Reusing the small CPU wave reference avoids
  bridging WebGPU and WebGL and gives height values directly to the Three.js mesh; the
  floor caustic is recomputed on the CPU and painted to a 2D canvas (a WebGPU canvas
  can't be sampled as a WebGL texture).
- **Symmetric wavemakers.** Pistons are placed symmetrically around the tank (an equal
  number per wall) and their layout is baked into the precomputed actuation asset.

## License

See repository settings.
