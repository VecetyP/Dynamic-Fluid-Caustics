// M6 · Caustic splat — spec render path (B): rasterised additive-blend splatting.
// One POINT primitive per surface cell. The vertex stage refracts a vertical
// light ray through the cell's surface normal (Snell), intersects the floor
// plane z = -d, and positions the point at that hit. The fragment stage emits a
// fixed energy that the pipeline's additive blend accumulates — race-free with
// no atomics (WGSL has none for textures), which is why path B suits WebGPU.

struct CParams {
  n         : u32,
  dx        : f32,
  d         : f32,   // focal distance surface → floor (m)
  nRel      : f32,   // n2/n1
  cellEnergy: f32,   // energy deposited per cell
  _pad0     : f32,
  _pad1     : f32,
  _pad2     : f32,
};

@group(0) @binding(0) var nTex : texture_2d<f32>;
@group(0) @binding(1) var<uniform> P : CParams;

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) energy    : f32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  var out: VSOut;
  let n = P.n;
  let x = vi % n;
  let y = vi / n;

  let nrm = textureLoad(nTex, vec2<i32>(i32(x), i32(y)), 0).xyz;

  // Vertical incoming ray; refract into the water.
  let incident = vec3<f32>(0.0, 0.0, -1.0);
  let refr = refract(incident, nrm, 1.0 / P.nRel);

  // Total internal reflection, or ray not heading down → cull offscreen.
  if (all(refr == vec3<f32>(0.0)) || refr.z >= -1e-4) {
    out.pos = vec4<f32>(2.0, 2.0, 0.0, 1.0);
    out.energy = 0.0;
    return out;
  }

  // Intersect floor plane at distance d below the surface.
  let t = P.d / (-refr.z);
  let extent = f32(n) * P.dx;
  let originXY = (vec2<f32>(f32(x), f32(y)) + 0.5) * P.dx;
  let hit = originXY + refr.xy * t;

  // Map floor hit (metres, [0,extent]) → clip space, y flipped.
  let uv = hit / extent;
  out.pos = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
  out.energy = P.cellEnergy;
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  // Accumulated as irradiance in .r; rgb kept equal for a neutral splat.
  return vec4<f32>(in.energy, in.energy, in.energy, in.energy);
}
