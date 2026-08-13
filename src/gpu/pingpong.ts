/**
 * Ping-pong storage textures — spec §5.1.
 *
 * The leapfrog stencil (eq 4.7) needs only h^n and h^{n-1} to produce h^{n+1},
 * so two buffers with an O(1) pointer swap suffice. We keep three r32float
 * textures in a ring: prev (h^{n-1}), curr (h^n), next (h^{n+1}); after each
 * step the roles rotate. All are STORAGE + TEXTURE_BINDING + COPY_SRC so the
 * same buffer feeds the M6 render pass without repacking.
 */

export interface PingPong {
  readonly size: number;
  /** h^{n-1} */
  prev: GPUTexture;
  /** h^n (also the texture handed to M6 as FluidState.heightTex) */
  curr: GPUTexture;
  /** h^{n+1} — write target of the current step */
  next: GPUTexture;
  /** Rotate prev <- curr <- next <- prev after a step is dispatched. */
  swap(): void;
  destroy(): void;
}

export function createHeightPingPong(device: GPUDevice, size: number): PingPong {
  const make = (label: string) =>
    device.createTexture({
      label,
      size: { width: size, height: size },
      format: "r32float",
      usage:
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_SRC |
        GPUTextureUsage.COPY_DST,
    });

  const ring: PingPong = {
    size,
    prev: make("h_prev"),
    curr: make("h_curr"),
    next: make("h_next"),
    swap() {
      const oldPrev = this.prev;
      this.prev = this.curr;
      this.curr = this.next;
      this.next = oldPrev;
    },
    destroy() {
      this.prev.destroy();
      this.curr.destroy();
      this.next.destroy();
    },
  };
  return ring;
}
