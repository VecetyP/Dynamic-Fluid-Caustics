/**
 * M-D · CausticPainter — CPU caustic onto a 2D canvas for the 3D tank floor.
 *
 * The 2D preview is rendered with WebGPU, and a WebGPU-backed canvas can't be
 * reliably sampled as a WebGL texture (Three reads it blank). So instead of
 * re-using that canvas, we recompute the caustic on the CPU from the same water
 * height field that drives the 3D surface, into an ordinary 2D canvas that WebGL
 * samples with no trouble.
 *
 * The transport mirrors M6 (`caustic.wgsl`): treat each surface sample as the
 * entry point of a vertical light ray, refract it through the local surface
 * normal, intersect the floor plane a distance `d` below, and deposit energy
 * there (bilinear splat). Where the surface focuses, rays pile up → bright
 * caustic; a flat surface spreads them evenly → uniform. Tone-mapped to a tinted
 * image. Cheap enough to run every frame at this resolution.
 */

export interface CausticParams {
  /** Focal distance surface → floor (m), matches the M6/main pipeline. */
  d: number;
  /** Relative refractive index n2/n1. */
  nRel: number;
  /** Sim cell size (m). */
  dx: number;
  /** Floor image resolution (texels per side). */
  res?: number;
  /** Rays per side cast from the surface (≥ res for a smooth splat). */
  rays?: number;
  /** Tone-map exposure. */
  exposure?: number;
}

export class CausticPainter {
  readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly img: ImageData;
  private readonly acc: Float32Array;
  private readonly res: number;
  private readonly rays: number;
  private readonly d: number;
  private readonly nRel: number;
  private readonly dx: number;
  private readonly exposure: number;

  constructor(p: CausticParams) {
    this.res = p.res ?? 160;
    this.rays = p.rays ?? 224;
    this.d = p.d;
    this.nRel = p.nRel;
    this.dx = p.dx;
    this.exposure = p.exposure ?? 0.55;

    this.canvas = document.createElement("canvas");
    this.canvas.width = this.canvas.height = this.res;
    this.ctx = this.canvas.getContext("2d")!;
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, this.res, this.res);
    this.img = this.ctx.createImageData(this.res, this.res);
    this.acc = new Float32Array(this.res * this.res);
  }

  /** Bilinear sample of the row-major n·n height field at grid coords (gx,gy). */
  private sampleH(field: Float32Array, n: number, gx: number, gy: number): number {
    if (gx < 0) gx = 0;
    else if (gx > n - 1) gx = n - 1;
    if (gy < 0) gy = 0;
    else if (gy > n - 1) gy = n - 1;
    const x0 = Math.floor(gx);
    const y0 = Math.floor(gy);
    const x1 = Math.min(n - 1, x0 + 1);
    const y1 = Math.min(n - 1, y0 + 1);
    const fx = gx - x0;
    const fy = gy - y0;
    const h00 = field[y0 * n + x0];
    const h10 = field[y0 * n + x1];
    const h01 = field[y1 * n + x0];
    const h11 = field[y1 * n + x1];
    return (h00 + (h10 - h00) * fx) * (1 - fy) + (h01 + (h11 - h01) * fx) * fy;
  }

  /** Recompute the floor caustic from the current surface. Pass null to clear. */
  paint(field: Float32Array | null, n: number): void {
    const res = this.res;
    const acc = this.acc;
    acc.fill(0);

    if (field && n > 1) {
      const rays = this.rays;
      const extent = n * this.dx;
      const eta = 1 / this.nRel;
      const eps = 0.5; // grid-cells, for slope central difference
      const energy = (res * res) / (rays * rays); // ⇒ flat surface accumulates ~1/texel

      for (let j = 0; j < rays; j++) {
        const v = (j + 0.5) / rays;
        const gy = v * (n - 1);
        for (let i = 0; i < rays; i++) {
          const u = (i + 0.5) / rays;
          const gx = u * (n - 1);

          // Surface slopes in physical units → normal (-hx,-hy,1) normalised.
          const hx =
            ((this.sampleH(field, n, gx + eps, gy) - this.sampleH(field, n, gx - eps, gy)) /
              (2 * eps)) /
            this.dx;
          const hy =
            ((this.sampleH(field, n, gx, gy + eps) - this.sampleH(field, n, gx, gy - eps)) /
              (2 * eps)) /
            this.dx;
          const nlen = Math.hypot(hx, hy, 1);
          const nx = -hx / nlen;
          const ny = -hy / nlen;
          const nz = 1 / nlen;

          // Refract the vertical incident ray I=(0,0,-1) through the surface.
          const cosi = nz; // -dot(N, I) = -(-nz) = nz
          const k = 1 - eta * eta * (1 - cosi * cosi);
          if (k < 0) continue; // total internal reflection
          const s = eta * cosi - Math.sqrt(k);
          const rx = eta * 0 + s * nx;
          const ry = eta * 0 + s * ny;
          const rz = eta * -1 + s * nz;
          if (rz >= -1e-4) continue; // not heading down → skip

          const t = this.d / -rz;
          const hitU = (u * extent - rx * t) / extent;
          const hitV = (v * extent - ry * t) / extent;

          // Bilinear splat into the floor accumulation buffer.
          const px = hitU * (res - 1);
          const py = hitV * (res - 1);
          if (px < 0 || py < 0 || px > res - 1 || py > res - 1) continue;
          const x0 = Math.floor(px);
          const y0 = Math.floor(py);
          const x1 = Math.min(res - 1, x0 + 1);
          const y1 = Math.min(res - 1, y0 + 1);
          const wx = px - x0;
          const wy = py - y0;
          acc[y0 * res + x0] += energy * (1 - wx) * (1 - wy);
          acc[y0 * res + x1] += energy * wx * (1 - wy);
          acc[y1 * res + x0] += energy * (1 - wx) * wy;
          acc[y1 * res + x1] += energy * wx * wy;
        }
      }
    }

    // Tone-map to a cool-white tint on black.
    const data = this.img.data;
    const exposure = this.exposure;
    for (let idx = 0; idx < acc.length; idx++) {
      const b = 1 - Math.exp(-exposure * acc[idx]);
      const o = idx * 4;
      data[o] = b * 180;
      data[o + 1] = b * 225;
      data[o + 2] = b * 255;
      data[o + 3] = 255;
    }
    this.ctx.putImageData(this.img, 0, 0);
  }
}
