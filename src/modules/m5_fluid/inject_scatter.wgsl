// M5 · Piston injection scatter. Writes the current step's per-piston amplitudes
// into the injection texture at each piston's cell. Non-piston cells are never
// written (the texture is zero-initialised and they stay zero), so the main wave
// step adds a nonzero value only at piston cells. One thread per piston.

struct SParams {
  p    : u32,   // number of pistons P
  t    : u32,   // schedule length T
  step : u32,   // current playback step (column into the schedule)
  n    : u32,   // grid dim
};

@group(0) @binding(0) var injTex : texture_storage_2d<r32float, write>;
@group(0) @binding(1) var<storage, read> cells : array<u32>; // piston grid indices
@group(0) @binding(2) var<storage, read> amps  : array<f32>; // amplitude[P][T], row-major
@group(0) @binding(3) var<uniform> S : SParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= S.p) { return; }
  let cell = cells[k];
  let xy = vec2<i32>(i32(cell % S.n), i32(cell / S.n));
  let amp = amps[k * S.t + S.step];
  textureStore(injTex, xy, vec4<f32>(amp, 0.0, 0.0, 0.0));
}
