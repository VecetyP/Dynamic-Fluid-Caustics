// M5 · Clear a r32float storage texture to zero. Used to reset the height
// ping-pong to rest before replaying a PistonSchedule from t=0.

struct CParams { n : u32 };

@group(0) @binding(0) var tex : texture_storage_2d<r32float, write>;
@group(0) @binding(1) var<uniform> P : CParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.n || gid.y >= P.n) { return; }
  textureStore(tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(0.0, 0.0, 0.0, 0.0));
}
