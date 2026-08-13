// M6 · Tone pass — spec POST note. Resolves the raw accumulated floor energy to
// a displayable image: exposure scale + exponential tone map, with a subtle
// water tint. Full-screen triangle, no vertex buffer.

struct TParams { exposure: f32 };

@group(0) @binding(0) var accum   : texture_2d<f32>;
@group(0) @binding(1) var samp    : sampler;
@group(0) @binding(2) var<uniform> P : TParams;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  // Oversized triangle covering the viewport.
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 3.0,  1.0),
  );
  return vec4<f32>(p[vi], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let dims = vec2<f32>(textureDimensions(accum));
  let uv = frag.xy / dims;
  let raw = textureSampleLevel(accum, samp, uv, 0.0).r;

  // Exponential tone map keeps bright convergence lines from clipping hard.
  let lit = 1.0 - exp(-raw * P.exposure);
  let water = vec3<f32>(0.05, 0.12, 0.16);
  let caustic = vec3<f32>(0.65, 0.9, 1.0);
  let col = mix(water, caustic, lit);
  return vec4<f32>(col, 1.0);
}
