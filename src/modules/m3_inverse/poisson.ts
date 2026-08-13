/**
 * DCT-based Neumann Poisson solver — TS port of the validated NumPy prototype
 * (`prototypes/m3_poisson/poisson_caustic.py`).
 *
 * Solves ∇²u = rhs with homogeneous Neumann BCs. The constant-coefficient
 * Laplacian on a regular grid diagonalises under the DCT-II, whose basis vectors
 * are its eigenvectors; we transform, divide by the stencil eigenvalues, and
 * invert. Framework-free (no WebGPU) so it runs the on-demand solve on the CPU
 * and is unit-testable under Node.
 *
 * Performance: a direct O(N³) separable transform via a precomputed cosine
 * matrix — a few ms at 128². Fast enough for an on-demand (per-sketch) solve;
 * swap in an FFT-based DCT later if the grid grows.
 */

/** Orthonormal DCT-II matrix F (n×n), row-major. F[k*n + m]. */
function buildDctMatrix(n: number): Float64Array {
  const F = new Float64Array(n * n);
  const a0 = Math.sqrt(1 / n);
  const ak = Math.sqrt(2 / n);
  for (let k = 0; k < n; k++) {
    const alpha = k === 0 ? a0 : ak;
    for (let m = 0; m < n; m++) {
      F[k * n + m] = alpha * Math.cos((Math.PI * (2 * m + 1) * k) / (2 * n));
    }
  }
  return F;
}

/** Apply a 1-D transform (matrix M, y[k]=Σ_m M[k,m] x[m]) along every row, then
 *  every column, of an n×n array. Separable ⇒ order is irrelevant. In-place. */
function transform2D(data: Float64Array, n: number, M: Float64Array): void {
  const tmp = new Float64Array(n);

  // Rows.
  for (let j = 0; j < n; j++) {
    const base = j * n;
    for (let k = 0; k < n; k++) {
      let s = 0;
      const mk = k * n;
      for (let m = 0; m < n; m++) s += M[mk + m] * data[base + m];
      tmp[k] = s;
    }
    for (let k = 0; k < n; k++) data[base + k] = tmp[k];
  }

  // Columns.
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < n; k++) {
      let s = 0;
      const mk = k * n;
      for (let m = 0; m < n; m++) s += M[mk + m] * data[m * n + i];
      tmp[k] = s;
    }
    for (let k = 0; k < n; k++) data[k * n + i] = tmp[k];
  }
}

/** Transpose of an n×n row-major matrix (the inverse of an orthonormal DCT). */
function transpose(M: Float64Array, n: number): Float64Array {
  const T = new Float64Array(n * n);
  for (let k = 0; k < n; k++)
    for (let m = 0; m < n; m++) T[m * n + k] = M[k * n + m];
  return T;
}

export interface PoissonSolver {
  readonly n: number;
  /** Solve ∇²u = rhs (row-major, length n*n). Returns u (length n*n). */
  solve(rhs: Float64Array, dx: number): Float64Array;
}

/** Build a reusable solver for a fixed grid size (caches the DCT matrices). */
export function makePoissonSolver(n: number): PoissonSolver {
  const F = buildDctMatrix(n);
  const Ft = transpose(F, n);

  // Eigenvalues of the 5-point Laplacian for DCT-II modes (per axis, Δx==Δy):
  //   λ_i = 2(cos(π i / n) − 1) / dx²  — dx factored out, applied at solve time.
  const lamUnit = new Float64Array(n);
  for (let i = 0; i < n; i++) lamUnit[i] = 2 * (Math.cos((Math.PI * i) / n) - 1);

  return {
    n,
    solve(rhs: Float64Array, dx: number): Float64Array {
      if (rhs.length !== n * n) throw new Error(`rhs must be length ${n * n}`);
      const hat = Float64Array.from(rhs);
      transform2D(hat, n, F); // forward DCT

      const invDx2 = 1 / (dx * dx);
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) {
          const denom = (lamUnit[j] + lamUnit[i]) * invDx2;
          const idx = j * n + i;
          // λ(0,0)=0 is the free additive constant → set DC mode to zero.
          hat[idx] = idx === 0 ? 0 : hat[idx] / denom;
        }
      }

      transform2D(hat, n, Ft); // inverse DCT
      return hat;
    },
  };
}
