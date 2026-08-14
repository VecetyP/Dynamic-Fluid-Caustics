# How It Works — Under the Hood

A deep dive for the curious: the math and engineering behind turning a drawing into
a focused water caustic, and back. This document is self-contained — you don't need
any external spec to follow it. It's kept up to date as the project evolves.

If you just want to run it, see the [README](./README.md). If you want to know *why*
every piece is the way it is, read on.

---

## The problem, stated plainly

A caustic is the bright pattern light makes after passing through a curved refracting
surface — the dancing web at the bottom of a swimming pool. Normally you have a water
surface and you get whatever caustic falls out of it. We want the inverse: **given a
target picture, find the water surface (and the wall motions that create it) whose
caustic reconstructs that picture on the tank floor.**

That's three nested inverse problems:

1. **Optics inverse** — what surface shape refracts light into the target image?
2. **Actuation inverse** — what piston motions make the water *become* that shape at a
   chosen instant?
3. **Forward check** — simulate the water for real and render the caustic to confirm.

The whole system is a chain of modules (`m1`…`m8`) that each own one step and talk
only through typed payloads:

```
draw ─▶ density ─▶ inverse surface ─▶ piston schedule ─▶ wave sim ─▶ caustic
 M1       M2            M3                  M4              M5         M6
                                     orchestrated by M7 · shown in 3D by M8
```

---

## 1. The forward model: shallow-water waves

The water surface height `h(x, y, t)` obeys the **linear wave equation** (the
shallow-water / small-amplitude limit):

```
∂²h/∂t² = c² ∇²h − γ ∂h/∂t
```

where `c = √(g·H)` is the wave speed (`g` gravity, `H` still-water depth) and `γ` is a
damping coefficient standing in for viscosity.

We discretize on a regular grid with an explicit **leapfrog** step:

```
hⁿ⁺¹ = ( 2hⁿ + (½γΔt − 1)hⁿ⁻¹ + c²Δt² ∇²hⁿ ) / (1 + ½γΔt)
```

with a 5-point Laplacian. Two facts fall out of this and are enforced in code:

- **Stability (CFL).** An explicit scheme blows up if the timestep is too large. The
  2D bound is `Δt ≤ Δx / (c√2)`; we run at `0.9×` that and *assert* it at startup
  (`physics.ts::assertCflStable`) — the sim refuses to run rather than produce
  garbage.
- **Reflective walls.** Neighbour reads are clamped at the grid edge, which imposes a
  zero-gradient (Neumann, `∂h/∂n = 0`) boundary — waves bounce off the glass instead
  of leaking out. It's branch-free (just index clamping).

The same integrator exists twice: once in WGSL on the GPU (`m5_fluid/*.wgsl`) and once
in plain TypeScript (`m5_fluid/reference.ts`). The CPU version is the **golden oracle**
the GPU shader is tested against.

---

## 2. How a surface focuses light

Take a vertical light ray hitting the water at point `x`. It refracts through the
local surface normal (Snell's law) and travels down to the floor a distance `d` below.
Where neighbouring rays *converge*, energy piles up — that's a bright caustic line;
where they spread, it's dim.

In the **paraxial** (gentle-slope) limit, a surface with height field `u` maps a floor
point by roughly `x ↦ x + ∇u` (scaled by `d` and the refractive index). The brightness
is the inverse of how much that map stretches area — i.e. `1 / det(I + D²u)`, the
Jacobian of the ray map, where `D²u` is the Hessian of the surface. Bright where the
surface is concave (a converging lens), dim where convex.

Linearizing `det(I + D²u) ≈ 1 + ∇²u` connects the target brightness `I` directly to the
**Laplacian** of the surface — which is exactly what makes the inverse solvable as a
Poisson problem.

---

## 3. The optics inverse (M3): a Poisson solve

We want a surface whose caustic has irradiance `I(x, y)` (the blurred drawing). From
the linearized relation above, the required "potential" `u` satisfies **Poisson's
equation**

```
∇²u = 1 − I / Ī
```

(`Ī` = mean of `I`, so the right-hand side is zero-mean — a solvability condition).
With reflective walls this is a **Neumann** boundary-value problem, and the clean way
to solve it is the **Discrete Cosine Transform**: the DCT diagonalizes the Laplacian
under Neumann conditions, so the solve is just "transform → divide by eigenvalues →
inverse transform." Implemented framework-free in `m3_inverse/poisson.ts`, validated
against a NumPy reference to a residual of ~1e-13.

The surface height is then `h_t = −u / [d(n_rel − 1)]` (from the refraction geometry;
`n_rel` is the water/air index ratio ≈ 1.333).

**Why targets must be low-contrast.** This is a *linear* (paraxial) approximation of a
genuinely nonlinear optics problem (the real relation is Monge–Ampère,
`det(I + D²u) = 1/I`). High-contrast targets need large surface curvature, where the
linearization breaks down. So `m2_density` deliberately keeps contrast low (bright
ambient base + a modest drawing gain) to stay in the regime where M3 is accurate.

---

## 4. The actuation inverse (M4): from a shape to piston motions

Now we have a *static* target surface `h_t`. But we can't just conjure it — we can only
push water from pistons on the walls. Because the wave equation is **linear**, the
surface at focal time `T` is a superposition of the responses to each piston's
emissions over time. Let `b_k(x, t)` be the surface response to a unit impulse from
piston `k`. Then an actuation where piston `k` emits amplitude `a_k(τ)` at time `τ`
gives

```
h(x, T) = Σ_k Σ_τ a_k(τ) · b_k(x, T − τ)
```

Flatten the surface samples into a vector and the `(piston, time)` pairs into another,
and this is one matrix equation:

```
M a = h_t          M ∈ ℝ^{N × (P·T)}
```

`M` (the **wave-basis matrix**) depends only on tank geometry, not on the drawing, so
we build it once by simulating one impulse per piston. It's tall, rectangular, and
horribly ill-conditioned (condition number ~1e19), so we don't invert it directly. We
solve a **Tikhonov-regularized least squares**:

```
a* = argmin ‖M a − h_t‖² + λ‖a‖²    ⇒    (MᵀM + λI) a* = Mᵀ h_t
```

The regularizer `λ‖a‖²` penalizes physically implausible high-energy actuations and
tames the conditioning. There's a lovely intuition here: `Mᵀ h_t` is the **adjoint /
time-reversal** of the target (play the target backwards through the tank) — already
recognizable on its own — and `(MᵀM + λI)⁻¹` is the correction that sharpens that
time-reversal guess into the true least-squares inverse.

Because `M⁺ = (MᵀM + λI)⁻¹ Mᵀ` depends only on geometry, it's **precomputed offline**
(`prototypes/m4_actuation/export_pinv.py`) and shipped as a JSON asset. At runtime,
turning any surface into a piston schedule is a single matrix–vector product
`a = M⁺ · h_t` (`m4_actuation/index.ts`). The schedule is time-reversed, so replaying
it forward makes the wavefronts converge onto `h_t` at time `T`.

---

## 5. The one rule that makes the loop close

There's a subtle trap: the basis matrix `M`, the GPU sim, and the CPU oracle must all
agree on *exactly* how a piston injects energy, down to the ordering. The canonical
rule everywhere is:

> **Advance the free leapfrog one step, THEN additively inject each piston's amplitude
> into the new surface at its cell.**

Because `M` is built from this exact operator, `M · a` equals a real forward replay of
`a` to machine precision (~1e-15). This is checked directly, and the full
`target → M4 → forward-sim → target` loop is a headless test
(`m4_actuation/closed_loop.test.ts`) that reconstructs the target to a few percent
error. Break the operator consistency and the loop silently stops closing — hence the
paranoia about keeping the three implementations identical.

---

## 6. Rendering the caustic (M6)

Given a surface, we render its caustic by **forward ray splatting**. For each surface
cell we refract a downward ray through the surface normal, intersect the floor plane a
distance `d` below, and deposit energy there:

```
hit = origin_xy − refr_xy · (d / −refr_z)
```

(The minus sign matters: with `h_t = −u/[d(n_rel−1)]` the physical transport is
`x + ∇u`, i.e. `origin − refr·t`. The `+` version renders the photographic negative.)

The classic way to accumulate overlapping splats is an atomic scatter-add, but **WGSL
has no float/texture atomics**, so that doesn't port. Instead we use **additive-blend
point splatting** (render path "B"): draw one point per ray with additive blending, and
the hardware blender does the accumulation, race-free. The caustic is rendered at a
resolution far higher than the sim grid by bilinearly sampling the surface normals —
the surface is smooth, so upsampling is faithful and you get a crisp caustic from a
coarse simulation.

---

## 7. Orchestration and timing (M7)

A small state machine (`INTERACTIVE / PULSE / IDLE`) drives everything. In **Pulse**
mode it replays a schedule from rest, holds the focused surface for a beat so the
caustic reads as a steady image, then resets and re-pulses. Pacing is **wall-clock**
(seconds, in `playback_timing.ts`), shared between the 2D and 3D views so they build,
focus, hold, and re-pulse in lockstep and run identically regardless of display refresh
rate.

---

## 8. The interactive 3D tank (M8)

The 3D environment is a **presentation layer** on top of the verified pipeline — it
adds no new physics.

- **Water on the CPU.** Rather than bridge raw WebGPU textures into WebGL, the 3D view
  reuses the tiny CPU wave reference to replay the schedule and get height values
  directly in JavaScript. At the small solver grid this is essentially free. Those
  heights displace a subdivided Three.js water mesh (bilinearly upsampled for
  smoothness), with per-step **linear interpolation** so motion is smooth and
  frame-rate independent.
- **Wavemakers.** Each perimeter piston is drawn as a paddle that translates in/out
  along its wall normal by the amplitude it's injecting that step — the intuitive
  "piston-type wavemaker." (The sim itself only prescribes a surface disturbance at the
  boundary cell; the paddle is a faithful visual proxy for that forcing, not a two-way
  rigid-body coupling.) Pistons are placed **symmetrically** — an equal count per wall
  at identical fractions — and that layout is baked into the actuation asset so the
  paddles line up with where the physics actually injects.
- **Floor caustic.** The caustic on the tank floor is recomputed on the CPU from the
  same water field (`m8_stage3d/floor_caustic.ts`), mirroring the M6 transport, and
  painted to a 2D canvas. Why not reuse the 2D WebGPU caustic directly? **A
  WebGPU-backed canvas can't be reliably sampled as a WebGL texture** — Three reads it
  blank — so the floor gets its own CPU pass.
- **Look.** Water uses a procedural, seamlessly-tiling ripple normal map plus
  image-based reflections (a PMREM'd gradient environment) so the wave slopes catch
  light and read as a surface.

---

## 9. How it's verified

There's no headless WebGPU in CI, so correctness is pinned with **golden-oracle**
tests: a CPU reference is the source of truth, and every solver has a NumPy prototype
and a TypeScript test that mirror each other. The strongest checks are end-to-end and
GPU-free:

- the inverse→forward loop reconstructs the target (`closed_loop.test.ts`);
- the 3D player's focal surface equals the forward-playback oracle **bit-for-bit**;
- the piston amplitudes exposed to the 3D paddles equal the schedule columns.

Anything genuinely visual (the actual WebGPU/WebGL render) is confirmed by eye in the
browser.

---

## 10. Deviations and known limits

- **Paraxial, not Monge–Ampère.** M3 solves the linearized optics. Great for smooth,
  low-contrast targets; a full nonlinear Monge–Ampère solve would handle high contrast
  and is a natural future upgrade.
- **Additive splatting instead of atomic scatter** — forced by WGSL's lack of atomics
  (see §6).
- **Small actuation asset.** The shipped `M⁺` is built for a small tank so it fits as
  inline JSON. A production-resolution `M⁺` is large and would ship as a compressed /
  streamed binary instead.
- **The 3D floor caustic is a CPU re-derivation**, not the same pixels as the WebGPU
  2D view (see §8) — they match in spirit and physics, not bit-for-bit.

---

*Questions or corrections welcome — open an issue.*
