// M6 · Tone pass — spec POST note. Resolves the raw accumulated floor energy to
// a displayable image: exposure scale + exponential tone map, with a subtle
// water tint. Full-screen triangle, no vertex buffer.
//
// UV is emitted per-vertex as a 0..1 coordinate across the visible quad, so the
// mapping is independent of BOTH the accumulator resolution and the swap-chain
// resolution. (Dividing frag coords by textureDimensions(accum) was the bug that
// crammed the image into the top-left 128/512 of the canvas.)

struct TParams { exposure: f32 };

@group(0) @binding(0) var accum   : texture_2d<f32>;
@group(0) @binding(1) var samp    : sampler;
@group(0) @binding(2) var<uniform> P : TParams;

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv        : vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  // Oversized triangle covering the viewport.
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 3.0,  1.0),
  );
  let clip = p[vi];
  var out: VSOut;
  out.pos = vec4<f32>(clip, 0.0, 1.0);
  // clip → uv: x∈[-1,1]→[0,1]; y flipped so screen-top = accum row 0.
  out.uv = clip * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
  return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let raw = textureSampleLevel(accum, samp, in.uv, 0.0).r;

  // Exponential tone map keeps bright convergence lines from clipping hard.
  let lit = 1.0 - exp(-raw * P.exposure);
  let water = vec3<f32>(0.05, 0.12, 0.16);
  let caustic = vec3<f32>(0.65, 0.9, 1.0);
  let col = mix(water, caustic, lit);
  return vec4<f32>(col, 1.0);
}
