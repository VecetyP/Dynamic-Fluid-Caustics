/**
 * M1 · Input Processing Canvas (spec Stage 1 / §3.1).
 *
 * A 2D drawing surface: the user paints a bright target on black. On demand it
 * rasterises the strokes and downsamples to the solver grid as a greyscale
 * intensity field for M2. Kept UI-only; the density math lives in M2.
 */

export class DrawingCanvas {
  private readonly ctx: CanvasRenderingContext2D;
  private drawing = false;
  private last: { x: number; y: number } | null = null;
  private readonly sampler: HTMLCanvasElement;

  constructor(private readonly canvas: HTMLCanvasElement, private readonly brush = 10) {
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("2D context unavailable for the drawing canvas.");
    this.ctx = ctx;
    this.sampler = document.createElement("canvas");
    this.clear();

    canvas.addEventListener("pointerdown", (e) => {
      this.drawing = true;
      this.last = this.pos(e);
      this.dot(this.last);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!this.drawing) return;
      const p = this.pos(e);
      this.stroke(this.last!, p);
      this.last = p;
    });
    const stop = () => {
      this.drawing = false;
      this.last = null;
    };
    canvas.addEventListener("pointerup", stop);
    canvas.addEventListener("pointerleave", stop);
  }

  private pos(e: PointerEvent): { x: number; y: number } {
    const r = this.canvas.getBoundingClientRect();
    return {
      x: ((e.clientX - r.left) / r.width) * this.canvas.width,
      y: ((e.clientY - r.top) / r.height) * this.canvas.height,
    };
  }

  private dot(p: { x: number; y: number }): void {
    this.ctx.fillStyle = "#fff";
    this.ctx.beginPath();
    this.ctx.arc(p.x, p.y, this.brush / 2, 0, Math.PI * 2);
    this.ctx.fill();
  }

  private stroke(a: { x: number; y: number }, b: { x: number; y: number }): void {
    this.ctx.strokeStyle = "#fff";
    this.ctx.lineWidth = this.brush;
    this.ctx.lineCap = "round";
    this.ctx.beginPath();
    this.ctx.moveTo(a.x, a.y);
    this.ctx.lineTo(b.x, b.y);
    this.ctx.stroke();
  }

  clear(): void {
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }

  /** True if anything has been drawn (any non-black pixel). */
  hasContent(): boolean {
    const { width: w, height: h } = this.canvas;
    const data = this.ctx.getImageData(0, 0, w, h).data;
    for (let i = 0; i < data.length; i += 4) if (data[i] > 8) return true;
    return false;
  }

  /** Downsample to an n×n greyscale intensity field in [0,1], row-major. */
  sampleIntensity(n: number): Float32Array {
    this.sampler.width = n;
    this.sampler.height = n;
    const sctx = this.sampler.getContext("2d", { willReadFrequently: true })!;
    sctx.clearRect(0, 0, n, n);
    sctx.drawImage(this.canvas, 0, 0, n, n); // area-averaged downsample
    const data = sctx.getImageData(0, 0, n, n).data;
    const out = new Float32Array(n * n);
    for (let i = 0; i < n * n; i++) out[i] = data[i * 4] / 255; // white → luminance
    return out;
  }
}
