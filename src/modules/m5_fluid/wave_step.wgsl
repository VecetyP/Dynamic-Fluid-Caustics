// M5 · Fluid Simulation Engine — damped shallow-water wave, explicit leapfrog.
// Implements spec eq 4.7:
//   h^{n+1} = [ 2h^n + (½γΔt − 1)h^{n-1} + c²Δt² L[h^n] ] / (1 + ½γΔt)
// One invocation per surface cell. Reflective (Neumann) walls via clamped
// neighbour loads. An optional Gaussian "poke" is injected into h^{n+1} so the
// user can excite ripples with no extra pass and no read-write storage texture.

struct SimParams {
  n      : u32,   // grid cells per side
  dx     : f32,   // cell size Δx (== Δy)
  c2dt2  : f32,   // c²·Δt²
  damp   : f32,   // ½·γ·Δt
  pokeX  : f32,   // poke centre, grid coords (px). pokeAmp==0 => no poke
  pokeY  : f32,
  pokeR  : f32,   // poke radius, cells
  pokeAmp: f32,   // poke displacement, metres
};

@group(0) @binding(0) var hCurr : texture_2d<f32>;              // h^n  (sampled)
@group(0) @binding(1) var hPrev : texture_2d<f32>;              // h^{n-1}
@group(0) @binding(2) var hNext : texture_storage_2d<r32float, write>; // h^{n+1}
@group(0) @binding(3) var<uniform> P : SimParams;

// Clamped integer load — clamping the coordinate mirrors the edge cell onto its
// ghost, giving ∂h/∂n = 0 (reflective wall) with no branching.
fn loadClamped(x: i32, y: i32) -> f32 {
  let m = i32(P.n) - 1;
  let cx = clamp(x, 0, m);
  let cy = clamp(y, 0, m);
  return textureLoad(hCurr, vec2<i32>(cx, cy), 0).r;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (gid.x >= P.n || gid.y >= P.n) { return; }

  let hc = textureLoad(hCurr, vec2<i32>(x, y), 0).r;
  let hp = textureLoad(hPrev, vec2<i32>(x, y), 0).r;

  // Five-point Laplacian (eq 4.5).
  let lap = (loadClamped(x + 1, y) + loadClamped(x - 1, y)
           + loadClamped(x, y + 1) + loadClamped(x, y - 1)
           - 4.0 * hc) / (P.dx * P.dx);

  // Leapfrog update (eq 4.7).
  let inv = 1.0 / (1.0 + P.damp);
  var hn = inv * (2.0 * hc + (P.damp - 1.0) * hp + P.c2dt2 * lap);

  // Optional Gaussian poke → additive displacement into h^{n+1}.
  if (P.pokeAmp != 0.0) {
    let dx = f32(x) - P.pokeX;
    let dy = f32(y) - P.pokeY;
    let r2 = dx * dx + dy * dy;
    hn = hn + P.pokeAmp * exp(-r2 / (2.0 * P.pokeR * P.pokeR));
  }

  textureStore(hNext, vec2<i32>(x, y), vec4<f32>(hn, 0.0, 0.0, 0.0));
}
