import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { ActuationMapper, type PinvMeta } from "./index.ts";
import meta from "./__fixtures__/pinv_medium.json";

// Resolved from the repo root (Vitest's cwd) to avoid import.meta.url, which
// trips a Vite SSR-transform bug.
const BIN_PATH = "src/modules/m4_actuation/__fixtures__/pinv_medium.bin";

// Guards the BINARY actuation asset + loader (ActuationMapper.fromBinary):
// reads the little-endian float32 .bin, builds the mapper, and checks the matvec
// on the golden sample reproduces the exported a* (catches endianness / shape /
// row-order bugs the inline-JSON path can't).

describe("M4 binary asset (32² medium)", () => {
  const m = meta as unknown as PinvMeta;

  it("fromBinary reproduces the exported golden actuation", () => {
    const buf = readFileSync(BIN_PATH);
    const pinv = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);

    const mapper = ActuationMapper.fromBinary(m.geometry, pinv);
    expect(mapper.rows).toBe(m.geometry.pistonCount * m.geometry.numSteps);
    expect(mapper.cols).toBe(m.geometry.n * m.geometry.n);

    const schedule = mapper.solve(Float32Array.from(m.sample!.hT));
    const expected = m.sample!.aExpected;
    expect(schedule.a.length).toBe(expected.length);

    let maxDiff = 0;
    for (let i = 0; i < expected.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(schedule.a[i] - expected[i]));
    }
    // float32 asset vs float64 golden → small but non-zero tolerance.
    expect(maxDiff).toBeLessThan(1e-4);
  });
});
