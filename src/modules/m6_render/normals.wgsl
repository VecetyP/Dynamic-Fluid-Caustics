// M6 · Normal pass — surface normals via central differences on the heightmap
// (spec §7.3 step 1). One invocation per cell. Writes RGBA16F: xyz normal + w slope.

struct NParams { n: u32, dx: f32 };

@group(0) @binding(0) var hTex : texture_2d<f32>;
@group(0) @binding(1) var nOut : texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> P : NParams;

fn h(x: i32, y: i32) -> f32 {
  let m = i32(P.n) - 1;
  return textureLoad(hTex, vec2<i32>(clamp(x, 0, m), clamp(y, 0, m)), 0).r;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.n || gid.y >= P.n) { return; }
  let x = i32(gid.x);
  let y = i32(gid.y);

  let hx = (h(x + 1, y) - h(x - 1, y)) / (2.0 * P.dx);
  let hy = (h(x, y + 1) - h(x, y - 1)) / (2.0 * P.dx);
  let nrm = normalize(vec3<f32>(-hx, -hy, 1.0));
  let slope = length(vec2<f32>(hx, hy));

  textureStore(nOut, vec2<i32>(x, y), vec4<f32>(nrm, slope));
}
