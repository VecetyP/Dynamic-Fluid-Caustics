// M6 · Caustic splat — spec render path (B): rasterised additive-blend splatting.
//
// Renders at an independent resolution `res` (>> sim grid n): one point per
// render cell, its surface normal BILINEARLY sampled from the n² normal texture.
// The surface is smooth, so upsampling is faithful and gives a high-resolution
// caustic from a coarse sim — no larger M⁺ asset needed. Each point refracts a
// vertical light ray through the sampled normal, intersects the floor plane, and
// deposits energy there via additive blending (race-free, no atomics).

struct CParams {
  n         : u32,   // sim grid dim (for physical extent)
  res       : u32,   // render grid dim (points per side)
  dx        : f32,   // sim cell size (m)
  d         : f32,   // focal distance surface → floor (m)
  nRel      : f32,   // n2/n1
  cellEnergy: f32,   // energy per render point (pre-scaled for res)
  _pad0     : f32,
  _pad1     : f32,
};

@group(0) @binding(0) var nTex : texture_2d<f32>;
@group(0) @binding(1) var nSamp : sampler;
@group(0) @binding(2) var<uniform> P : CParams;

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) energy    : f32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var out: VSOut;
  let res = P.res;
  let x = vi % res;
  let y = vi / res;

  // Normalised entry point of this ray on the surface, and the bilinear normal.
  let uv = (vec2<f32>(f32(x), f32(y)) + 0.5) / f32(res);
  let nrm = textureSampleLevel(nTex, nSamp, uv, 0.0).xyz;

  let incident = vec3<f32>(0.0, 0.0, -1.0);
  let refr = refract(incident, nrm, 1.0 / P.nRel);
  if (all(refr == vec3<f32>(0.0)) || refr.z >= -1e-4) {
    out.pos = vec4<f32>(2.0, 2.0, 0.0, 1.0); // TIR / upward → cull offscreen
    out.energy = 0.0;
    return out;
  }

  let extent = f32(P.n) * P.dx;
  let originXY = uv * extent;
  let t = P.d / (-refr.z);
  // Subtract the refracted offset: with h_t = −u/[d(n_rel−1)] (eq 4.3) the
  // physical transport is x + ∇u = originXY − refr.xy·t.
  let hit = originXY - refr.xy * t;

  let huv = hit / extent;
  out.pos = vec4<f32>(huv.x * 2.0 - 1.0, 1.0 - huv.y * 2.0, 0.0, 1.0);
  out.energy = P.cellEnergy;
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  return vec4<f32>(in.energy, in.energy, in.energy, in.energy);
}
