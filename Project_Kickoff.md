# Dynamic Fluid Caustics — Project Kickoff

Stack: **WebGPU + TypeScript**. Approach: **foundation-first**, first milestone aimed at spec Phase 1 (manually-poked ripple casts a moving caustic).

This is the on-ramp from the technical spec (Rev 2.0) to running code. It reorders the spec's §8 plan slightly so the foundation gets exercised by a working render early instead of built in isolation.

## 0. Decisions to lock before coding

- **Grid resolution**: start at **128×128** (sim grid N), not 256². Everything is bandwidth-bound; 128² keeps frame time trivial and iteration fast. Bump to 256² once the pipeline is proven.
- **Units**: pick metres for `cell_size`, `d` (focal distance), and pin still-depth `H`. Write them down; the CFL bound (eq 4.8) depends on `c = √(gH)`.
- **Render path**: use **additive-blend splatting (spec path B), not atomic scatter (path A)**. WebGPU/WGSL has no float image atomics and no texture atomics — path A's `imageAtomicAdd` doesn't port cleanly. Path B (render one point per surface cell into a float target with additive blending) is the natural WebGPU structure and is race-free via the ROP. This is the single most important stack-specific deviation from the spec.
- **Numeric precision**: `f32` throughout (WebGPU has no f64). Fine for this problem.

## 1. Repo + tooling scaffold (day 1)

- `npm create vite@latest` → vanilla-ts template. Add `@webgpu/types`.
- Structure mirrors the spec's module decomposition (§3.1) so contracts stay honest:
  ```
  src/
    contracts/     # §3.2 typed payloads as TS interfaces (RawCanvas, DensityMap,
                   #   TargetHeightmap, PistonSchedule, FluidState, CausticBuffer)
    modules/
      m1_canvas/   m2_density/   m3_inverse/   m4_actuation/
      m5_fluid/    m6_render/    m7_orchestrator/
    gpu/           # device init, pipeline + bind-group helpers, ping-pong texture util
    test/          # golden-image regression harness
    main.ts
  ```
- Add **Vitest** now. The spec repeatedly calls for golden-heightmap regression tests; wiring the harness before the solvers exist means every module lands with a test.
- Feature-detect WebGPU and fail loudly with a clear message (Safari/Firefox flags vary).

## 2. Data contracts first (§3.2)

Transcribe the six structs into `contracts/` as TS interfaces before any logic. These are the module boundaries — no module reaches into another's state. Getting them fixed up front is what "foundation properly" buys you: M5/M6 (per-frame) and M3/M4 (on-demand) can then be built and tested independently against golden data.

Key ones to define immediately: `FluidState` (height R32F texture, normal RGBA16F, `cell_size`) and `CausticBuffer` (`floor_accum`, `exposure`) — these are the per-frame GPU-resident payloads M5→M6→screen.

## 3. GPU foundation (`gpu/`)

- Device/adapter init, canvas context config.
- **Ping-pong texture helper**: two `r32float` storage textures (`h^n`, `h^{n-1}`) with a pointer-swap each step (spec §5.1). This is the spine of M5.
- Thin helpers for compute pipeline + bind group creation so shader code stays readable.

## 4. Milestone 1 — M5 fluid sim (the real first build)

Implement the damped-wave leapfrog update (eq 4.7) as a WGSL compute shader, one invocation per cell, 16×16 workgroups:

- Five-point Laplacian (eq 4.5) reading `h^n`, writing `h^{n+1}`.
- Damping + `c²Δt²` terms per eq 4.7.
- **Assert the CFL bound at init**: `Δt ≤ Δx/(c√2)`; choose Δt at ~0.9× the bound (spec §4.2, §9). If violated, refuse to run.
- Boundaries: reflective Neumann via ghost cells (mirror interior). Defer piston Dirichlet forcing until M4 exists — for now, inject a manual "poke" (raise a few cells) to create a ripple.

**Test**: seed a single-cell impulse, step, and check symmetry + energy decay against a NumPy reference (golden array). This validates the stencil independent of rendering.

## 5. Milestone 1 — M6 refraction render

- **Normal pass**: compute shader, central differences on the heightmap → RGBA16F normal texture (spec §7.3 steps 1).
- **Caustic pass (path B)**: for each surface cell, refract the vertical incoming ray through the surface normal (Snell, `refract()` in WGSL), intersect the floor plane at `z = -d`, and splat the cell's energy at the hit point into a float accumulation target with additive blending. Render as instanced points (one per cell) — vertex shader computes the hit position, fragment outputs energy.
- **Tone pass**: full-screen quad divides `floor_accum` by ray count / exposure → displayable R16F, optional bloom (spec POST note).

**Acceptance (= spec Phase 1)**: poke the surface with the mouse, watch the ripple propagate, reflect off walls, and cast a *moving* caustic pattern on the floor. When you see that, the per-frame path (M5+M6) and the whole GPU foundation are proven.

## 6. What comes after (spec Phases 2–4, later)

- **Phase 2 — Math core**: M3 inverse-caustic as the **Poisson solve** (eq 4.2/4.3), not Monge–Ampère. On a regular grid this is one linear solve; an **FFT/DCT Poisson solver** is the sub-millisecond runtime path. Prototype it in Python/NumPy first (validate against a synthetic target heightmap), then port. M4 = least-squares actuation against the pre-computed pseudoinverse (eq 4.12).
- **Phase 3 — Piston automation**: feed `PistonSchedule` into M5's boundary forcing; calibrate γ, Δt, λ so the surface converges at focal time T. Add M7 state machine + Pulse Mode looping.
- **Phase 4 — UI**: M1 drawing canvas wired in, orbit camera, exposure + Pulse-Mode controls.

## First-week concrete target

Days 1–2: scaffold, contracts, GPU init, ping-pong util, Vitest harness.
Days 3–5: M5 compute shader + stencil test passing against NumPy golden.
Days 6–7: M6 normal + caustic + tone passes; manual poke → moving caustic on screen.

That gets you from spec-on-paper to a live, interactive caustic in about a week, with the module boundaries and test harness already load-bearing.
