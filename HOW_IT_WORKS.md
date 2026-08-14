# How It Works

This is the long version, for anyone who wants to know what's going on under the hood.
You don't need any outside document to follow it. I keep it up to date as the project
grows.

If you just want to run the thing, the [README](./README.md) has you covered.

## The problem, in plain terms

A caustic is the bright pattern light makes after passing through a curved surface,
like the shifting web at the bottom of a pool. Normally you have a surface and you get
whatever caustic falls out of it. Here we want the opposite. Given a target picture,
find the water surface whose caustic reconstructs that picture on the tank floor, and
find the wall motions that make the water take that shape.

That's really three inverse problems stacked on top of each other:

1. Optics: what surface shape refracts light into the target image?
2. Actuation: what piston motions make the water become that shape at a chosen instant?
3. Forward check: simulate the water for real and render the caustic to confirm it.

The code is a chain of modules (`m1` through `m8`), each owning one step, passing typed
payloads between them:

```
draw -> density -> inverse surface -> piston schedule -> wave sim -> caustic
 M1       M2            M3                  M4              M5         M6
                                    orchestrated by M7, shown in 3D by M8
```

## 1. The forward model: shallow-water waves

The water height `h(x, y, t)` follows the linear wave equation, which is the
small-amplitude, shallow-water limit:

```
d²h/dt² = c² ∇²h - γ dh/dt
```

Here `c = √(g·H)` is the wave speed (`g` gravity, `H` the resting depth), and `γ` is a
damping term that stands in for viscosity.

On a grid, we march it forward with an explicit leapfrog step:

```
h(n+1) = ( 2h(n) + (½γΔt - 1)h(n-1) + c²Δt² ∇²h(n) ) / (1 + ½γΔt)
```

with a 5-point Laplacian. Two things come straight out of this and both show up in the
code.

First, stability. An explicit scheme like this blows up if the timestep is too large.
The 2D bound is `Δt ≤ Δx / (c√2)`, and we run at 0.9 of that. The sim asserts the bound
at startup (`physics.ts`), so if the numbers are ever wrong it stops instead of
producing nonsense.

Second, the walls. Neighbour reads are clamped at the grid edge, which imposes a
zero-gradient (Neumann) boundary. Waves reflect off the glass instead of leaking out,
and it costs nothing but an index clamp.

This integrator exists twice: once in WGSL on the GPU (`m5_fluid/*.wgsl`), and once in
plain TypeScript (`m5_fluid/reference.ts`). The CPU version is the reference of record,
the thing the GPU shader gets tested against.

## 2. How a surface focuses light

Picture a vertical ray of light hitting the water at point `x`. It refracts through the
local surface normal (Snell's law) and travels down to the floor a distance `d` below.
Where neighbouring rays bunch together, energy piles up and you get a bright caustic
line. Where they spread apart, it goes dim.

In the paraxial limit (gentle slopes), a surface with height field `u` sends a floor
point to roughly `x + ∇u`, scaled by `d` and the refractive index. The brightness is
the inverse of how much that mapping stretches area, which is `1 / det(I + D²u)`, where
`D²u` is the Hessian of the surface. Bright where the surface is concave and acts like
a converging lens, dim where it's convex.

Linearising `det(I + D²u) ≈ 1 + ∇²u` ties the target brightness `I` directly to the
Laplacian of the surface, and that is what makes the inverse a Poisson problem.

## 3. The optics inverse (M3): a Poisson solve

We want a surface whose caustic has brightness `I(x, y)`, the blurred drawing. From the
relation above, the surface potential `u` satisfies

```
∇²u = 1 - I / Ī
```

where `Ī` is the mean of `I`, so the right side averages to zero (a condition the
problem needs to be solvable). With reflecting walls this is a Neumann boundary-value
problem, and the clean way to solve it is the Discrete Cosine Transform. The DCT
diagonalises the Laplacian under Neumann conditions, so the solve becomes: transform,
divide by the eigenvalues, transform back. It lives in `m3_inverse/poisson.ts`, checked
against a NumPy reference down to a residual around 1e-13.

The surface height then follows from the refraction geometry as
`h_t = -u / [d(n_rel - 1)]`, where `n_rel` is the water-to-air index ratio, about 1.333.

There's a catch, and it's worth being honest about it. This is a linear (paraxial)
approximation of a problem that's genuinely nonlinear. The true relation is
Monge-Ampère, `det(I + D²u) = 1/I`. High-contrast targets need large surface curvature,
which is exactly where the linear approximation falls apart. So `m2_density`
deliberately keeps contrast low (a bright ambient base plus a modest drawing gain), to
stay in the range where M3 is accurate.

## 4. The actuation inverse (M4): from a shape to piston motions

Now we have a still target surface `h_t`. We can't just will it into existence. All we
can do is push water from pistons on the walls. The saving grace is that the wave
equation is linear, so the surface at focal time `T` is a sum of the responses to each
piston's pushes over time. Write `b_k(x, t)` for the surface response to a single
impulse from piston `k`. Then if piston `k` emits amplitude `a_k(τ)` at time `τ`:

```
h(x, T) = Σ_k Σ_τ a_k(τ) · b_k(x, T - τ)
```

Flatten the surface samples into one vector and the (piston, time) pairs into another,
and this is a single matrix equation:

```
M a = h_t          M ∈ ℝ^(N × P·T)
```

`M`, the wave-basis matrix, depends only on the tank geometry, not on the drawing, so
we build it once by simulating one impulse per piston. It's tall, rectangular, and
badly conditioned (condition number around 1e19), so inverting it directly is a bad
idea. Instead we solve a Tikhonov-regularised least squares:

```
a* = argmin ‖M a - h_t‖² + λ‖a‖²    which gives    (MᵀM + λI) a* = Mᵀ h_t
```

The `λ‖a‖²` term penalises wildly high-energy actuations and tames the conditioning.
There's a nice picture hiding in here. `Mᵀ h_t` is the time-reversal of the target,
play the target backwards through the tank, and it's already recognisable on its own.
The `(MᵀM + λI)⁻¹` factor is the correction that sharpens that time-reversed guess into
the real least-squares answer.

Since `M⁺ = (MᵀM + λI)⁻¹ Mᵀ` depends only on geometry, we compute it offline
(`prototypes/m4_actuation/export_pinv.py`) and ship it as a JSON asset. At runtime,
turning a surface into a piston schedule is one matrix-vector product, `a = M⁺ · h_t`
(`m4_actuation/index.ts`). The schedule comes out time-reversed, so replaying it forward
makes the wavefronts converge onto `h_t` at time `T`.

## 5. The one rule that makes the loop close

Here's a subtle trap. The basis matrix `M`, the GPU sim, and the CPU reference all have
to agree on exactly how a piston injects energy, right down to the ordering. The rule
everywhere is this:

> Advance the free leapfrog one step, then add each piston's amplitude into the new
> surface at its cell.

Because `M` is built from that exact operator, `M · a` matches a real forward replay of
`a` to machine precision (around 1e-15). We check that directly, and the whole
`target -> M4 -> forward sim -> target` loop is a headless test
(`m4_actuation/closed_loop.test.ts`) that reconstructs the target to a few percent. If
you ever let the three implementations drift apart, the loop quietly stops closing,
which is why there's so much fuss about keeping them identical.

## 6. Rendering the caustic (M6)

Given a surface, we render its caustic by throwing rays forward. For each surface cell,
refract a downward ray through the surface normal, find where it hits the floor plane a
distance `d` below, and drop energy there:

```
hit = origin_xy - refr_xy · (d / -refr_z)
```

The minus sign is not cosmetic. With `h_t = -u/[d(n_rel-1)]`, the physical transport is
`x + ∇u`, which is `origin - refr·t`. Flip the sign and you render the photographic
negative.

The usual way to pile up overlapping splats is an atomic scatter-add, but WGSL has no
float or texture atomics, so that route is closed. We use additive-blend point
splatting instead: draw one point per ray with additive blending on, and the hardware
blender does the accumulation for us with no race. We render at a resolution much
higher than the sim grid by bilinearly sampling the surface normals. The surface is
smooth, so upsampling is faithful, and a coarse simulation still gives a crisp caustic.

## 7. Orchestration and timing (M7)

A small state machine (`INTERACTIVE`, `PULSE`, `IDLE`) runs the show. In Pulse mode it
replays a schedule from rest, holds the focused surface for a moment so the caustic
reads as a still image, then resets and pulses again. The pacing is wall-clock, in
seconds, defined in `playback_timing.ts` and shared between the 2D and 3D views. That
way they build, focus, hold, and re-pulse together, and they run at the same speed no
matter what your display's refresh rate is.

## 8. The interactive 3D tank (M8)

The 3D environment sits on top of the verified pipeline. It adds no new physics.

The water runs on the CPU. Rather than bridge raw WebGPU textures into WebGL, the 3D
view reuses the small CPU wave reference to replay the schedule and read height values
straight out in JavaScript. At this grid size that's basically free. Those heights
displace a subdivided Three.js water mesh, upsampled for smoothness, with per-step
linear interpolation so the motion stays smooth and independent of frame rate.

The wavemakers are the paddles around the rim. Each one slides in and out along its
wall normal by the amount its piston is injecting on that step, which is how a real
piston-type wavemaker behaves. Worth being clear here: the sim only prescribes a
surface disturbance at the boundary cell, so the paddle is an honest visual stand-in for
that forcing, not a two-way rigid-body coupling. The pistons are placed symmetrically,
the same count on every wall at the same positions, and that layout is baked into the
actuation asset so the paddles sit where the physics actually pushes.

The floor caustic is recomputed on the CPU from the same water field
(`m8_stage3d/floor_caustic.ts`), following the M6 transport, and painted onto a plain
2D canvas. Why not reuse the 2D WebGPU caustic directly? Because a WebGPU-backed canvas
can't be read reliably as a WebGL texture. Three sees it as blank, so the floor gets its
own CPU pass.

For the look, the water uses a procedural, seamlessly tiling ripple normal map plus
image-based reflections (a blurred gradient environment), so the wave slopes catch
light and actually read as a surface rather than a flat sheet.

## 9. How it's checked

There's no headless WebGPU in CI, so correctness leans on reference implementations. A
CPU version is the source of truth, and every solver has a NumPy prototype and a
TypeScript test that mirror each other. The strongest checks are end to end and don't
touch the GPU:

- the inverse-then-forward loop reconstructs the target (`closed_loop.test.ts`);
- the 3D player's focal surface equals the forward-playback reference, exactly;
- the piston amplitudes handed to the 3D paddles equal the schedule columns.

Anything that's genuinely visual (the real WebGPU or WebGL render) gets confirmed by
eye in the browser.

## 10. Where it cuts corners

M3 solves the linearised optics, not the full Monge-Ampère problem. That's great for
smooth, low-contrast targets, and a full nonlinear solve is the obvious next upgrade if
you want high contrast.

The caustic uses additive splatting instead of an atomic scatter, forced by WGSL's lack
of atomics (see section 6).

The shipped actuation asset is built for a small tank so it fits as inline JSON. A
full-resolution `M⁺` is large and would ship as a compressed or streamed binary instead.

The 3D floor caustic is a CPU re-derivation, not the same pixels as the 2D WebGPU view
(see section 8). They agree on physics, not on exact pixels.

Questions or corrections are welcome. Open an issue.
