import { describe, it, expect } from "vitest";
import { WaveFieldCPU } from "./reference.ts";
import {
  cflMaxDt,
  chooseDt,
  assertCflStable,
  DEFAULT_PARAMS,
  type WaveParams,
} from "../../physics.ts";

const P: WaveParams = { ...DEFAULT_PARAMS, n: 64 };
const idx = (n: number, x: number, y: number) => y * n + x;

describe("CFL stability (eq 4.8)", () => {
  it("accepts the chosen dt (0.9× bound)", () => {
    expect(() => assertCflStable(P, chooseDt(P))).not.toThrow();
  });

  it("rejects a dt above the CFL bound", () => {
    const tooBig = cflMaxDt(P) * 1.01;
    expect(() => assertCflStable(P, tooBig)).toThrow(/CFL violation/);
  });
});

describe("M5 leapfrog reference", () => {
  it("keeps a centred impulse symmetric under reflective walls", () => {
    const dt = chooseDt(P);
    const w = new WaveFieldCPU(P, dt);
    // True reflection axis of an n-grid is (n-1)/2; poking there keeps the
    // x <-> (n-1-x) mirror exact.
    const c = (P.n - 1) / 2;
    w.poke(c, c, 2, 1);
    for (let i = 0; i < 40; i++) w.step();

    // Symmetry across the vertical axis through the centre column.
    const n = P.n;
    let maxAsym = 0;
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n / 2; x++) {
        const l = w.curr[idx(n, x, y)];
        const r = w.curr[idx(n, n - 1 - x, y)];
        maxAsym = Math.max(maxAsym, Math.abs(l - r));
      }
    }
    expect(maxAsym).toBeLessThan(1e-5);
  });

  it("does not blow up and dissipates energy over time (damping)", () => {
    const dt = chooseDt(P);
    const w = new WaveFieldCPU(P, dt);
    w.poke(P.n / 2, P.n / 2, 2, 1);
    const e0 = w.sumSquares();
    let ePrev = e0;
    for (let block = 0; block < 5; block++) {
      for (let i = 0; i < 50; i++) w.step();
      const e = w.sumSquares();
      expect(Number.isFinite(e)).toBe(true);
      expect(e).toBeLessThanOrEqual(e0 * 1.001); // never gains energy
      ePrev = e;
    }
    expect(ePrev).toBeLessThan(e0); // strictly decayed after 250 steps
  });
});
