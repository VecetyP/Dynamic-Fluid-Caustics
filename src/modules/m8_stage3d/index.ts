/**
 * M8 · Stage3D — the interactive 3D tank (presentation layer).
 *
 * A NEW presentation layer on top of the verified 2D pipeline (M1–M7); it does
 * NOT touch the numerics. Three.js (WebGL) renders a glass tank with a textured
 * water surface, a floor that shows the live caustic, an overhead light + light
 * shaft, and an orbit camera.
 *
 * Plug-in points for the milestone track:
 *   M-B water displacement → `displaceWater()` mutates the water vertices.
 *   M-C pistons           → `buildPistons()` / `setPistonOffsets()` (symmetric).
 *   M-D caustic on floor   → `setFloorCaustic(canvas)` maps the 2D caustic canvas
 *                            onto the floor as an emissive texture.
 */

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export interface Stage3DConfig {
  /** Sim grid cells per side (matches the M⁺ asset geometry, e.g. 16). */
  gridN: number;
  /** Tank inner extent in X and Z, world units. */
  tankSize: number;
  /** Water-surface-to-floor distance, world units (the light's travel depth). */
  tankDepth: number;
  /** Glass wall thickness, world units. */
  wallThickness: number;
  /** Water mesh subdivisions per side (upsampled beyond gridN for a smooth
   *  surface once M-B displaces it). */
  waterSegments: number;
}

export const DEFAULT_STAGE_CONFIG: Stage3DConfig = {
  gridN: 16,
  tankSize: 2.0,
  tankDepth: 1.0,
  wallThickness: 0.04,
  waterSegments: 64,
};

export class Stage3D {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly renderer: THREE.WebGLRenderer;
  readonly controls: OrbitControls;
  readonly cfg: Stage3DConfig;

  /** Water surface (y = 0). M-B displaces its vertices. */
  readonly waterMesh: THREE.Mesh;
  /** Tank floor plane (y = -tankDepth). M-D projects the caustic here. */
  readonly floorMesh: THREE.Mesh;

  /** World Y per unit of height-field value (vertical exaggeration for M-B). */
  waterVerticalScale = 0.2;
  private waterU!: Float32Array;
  private waterV!: Float32Array;
  private waterNormalTex: THREE.Texture | null = null;

  /** World travel per unit of piston amplitude — paddles slide in/out (M-C). */
  pistonTravelScale = 0.15;
  private pistons: THREE.Mesh[] = [];
  private pistonBase: THREE.Vector3[] = [];
  /** Inward horizontal unit vector (in XZ) each paddle pushes along. */
  private pistonDir: THREE.Vector3[] = [];

  /** Live caustic texture drawn onto the floor (M-D). */
  private causticTex: THREE.CanvasTexture | null = null;

  private readonly container: HTMLElement;
  private readonly resizeObserver: ResizeObserver;
  private rafId = 0;
  private running = false;
  private elapsed = 0;
  /** Optional hook run every frame before render (drives the sim in M-B/M-C). */
  onFrame: ((dtSeconds: number) => void) | null = null;
  private lastTime = 0;

  constructor(canvas: HTMLCanvasElement, cfg: Partial<Stage3DConfig> = {}) {
    this.cfg = { ...DEFAULT_STAGE_CONFIG, ...cfg };
    this.container = (canvas.parentElement ?? canvas) as HTMLElement;

    const { tankSize: S, tankDepth: D } = this.cfg;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setClearColor(0x0b0e11, 1);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    this.camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
    this.camera.position.set(S * 1.15, D * 1.7, S * 1.35);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, -D * 0.35, 0);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = S * 0.6;
    this.controls.maxDistance = S * 6;
    // Full vertical orbit (no polar clamp): rise overhead or dip below the water.

    this.setupEnvironment(); // image-based reflections so wave slopes read
    this.buildLights();
    const { water, floor } = this.buildTank();
    this.waterMesh = water;
    this.floorMesh = floor;
    this.precomputeWaterUV();

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    this.resize();
  }

  // ---------------------------------------------------------------------------
  // Grid ↔ world mapping
  // ---------------------------------------------------------------------------

  /** Map a sim-grid cell (ix, iy ∈ [0, gridN-1]) to a world point on the water
   *  plane. Returns {x, z}; y is the surface (0) unless displaced. */
  gridToWorld(ix: number, iy: number): { x: number; z: number } {
    const { gridN, tankSize: S } = this.cfg;
    const u = gridN > 1 ? ix / (gridN - 1) : 0.5;
    const v = gridN > 1 ? iy / (gridN - 1) : 0.5;
    return { x: (u - 0.5) * S, z: (v - 0.5) * S };
  }

  /** The water surface geometry — M-B mutates position.y per vertex. */
  get waterGeometry(): THREE.PlaneGeometry {
    return this.waterMesh.geometry as THREE.PlaneGeometry;
  }

  private precomputeWaterUV(): void {
    const pos = this.waterGeometry.attributes.position;
    const S = this.cfg.tankSize;
    const count = pos.count;
    this.waterU = new Float32Array(count);
    this.waterV = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const x = pos.getX(i);
      const z = pos.getZ(i);
      this.waterU[i] = Math.min(1, Math.max(0, x / S + 0.5));
      this.waterV[i] = Math.min(1, Math.max(0, z / S + 0.5));
    }
  }

  /** Displace the water surface from a row-major n·n height field (bilinear
   *  upsample). Pass null to flatten. Recomputes normals so lighting follows the
   *  waves. */
  displaceWater(field: Float32Array | null, n: number): void {
    const geo = this.waterGeometry;
    const pos = geo.attributes.position;
    const scale = this.waterVerticalScale;
    for (let i = 0; i < pos.count; i++) {
      let y = 0;
      if (field && n > 1) {
        const gx = this.waterU[i] * (n - 1);
        const gy = this.waterV[i] * (n - 1);
        const x0 = Math.min(n - 1, Math.floor(gx));
        const y0 = Math.min(n - 1, Math.floor(gy));
        const x1 = Math.min(n - 1, x0 + 1);
        const y1 = Math.min(n - 1, y0 + 1);
        const fx = gx - x0;
        const fy = gy - y0;
        const h00 = field[y0 * n + x0];
        const h10 = field[y0 * n + x1];
        const h01 = field[y1 * n + x0];
        const h11 = field[y1 * n + x1];
        const top = h00 + (h10 - h00) * fx;
        const bot = h01 + (h11 - h01) * fx;
        y = (top + (bot - top) * fy) * scale;
      }
      pos.setY(i, y);
    }
    pos.needsUpdate = true;
    geo.computeVertexNormals();
  }

  /** Flatten the water surface (idle state). */
  flattenWater(): void {
    this.displaceWater(null, 0);
  }

  // ---------------------------------------------------------------------------
  // Pistons (M-C) — symmetric wavemaker paddles
  // ---------------------------------------------------------------------------

  /** Place a wavemaker PADDLE for each perimeter piston. Paddles are thin vertical
   *  plates flush to their wall that translate in/out along the wall's inward
   *  normal. When the piston count is a multiple of 4 (the symmetric asset case)
   *  they are placed at identical interior fractions per side, so every wall looks
   *  the same and the middle paddle sits dead-centre. `cells` are flat grid
   *  indices in `PistonSchedule` order (top, right, bottom, left); `n` is the sim
   *  grid. Idempotent. */
  buildPistons(cells: ArrayLike<number>, n: number): void {
    for (const p of this.pistons) {
      this.scene.remove(p);
      p.geometry.dispose();
    }
    this.pistons = [];
    this.pistonBase = [];
    this.pistonDir = [];

    const { tankSize: S, tankDepth: D } = this.cfg;
    const half = S / 2;
    const cellW = n > 1 ? S / (n - 1) : S;
    const plateW = cellW * 0.9; // span along the wall
    const plateT = cellW * 0.3; // thickness (inward)
    const plateH = D * 0.8; // spans most of the water column
    const baseY = D * 0.06 - plateH / 2; // top just above the still water line
    const inset = plateT * 0.5;

    const mat = new THREE.MeshStandardMaterial({
      color: 0xb2c0d0,
      metalness: 0.75,
      roughness: 0.3,
      emissive: 0x0a1016,
    });

    const P = cells.length;
    const addPaddle = (
      x: number,
      z: number,
      dx: number,
      dz: number,
      sizeX: number,
      sizeZ: number
    ): void => {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(sizeX, plateH, sizeZ), mat);
      const base = new THREE.Vector3(x + dx * inset, baseY, z + dz * inset);
      mesh.position.copy(base);
      this.scene.add(mesh);
      this.pistons.push(mesh);
      this.pistonBase.push(base);
      this.pistonDir.push(new THREE.Vector3(dx, 0, dz));
    };

    if (P % 4 === 0) {
      // Symmetric placement: perSide paddles at fractions (j+1)/(perSide+1) along
      // each wall, in the same side order the asset generates (top,right,bottom,left).
      const perSide = P / 4;
      for (let s = 0; s < 4; s++) {
        for (let j = 0; j < perSide; j++) {
          const tang = ((j + 1) / (perSide + 1) - 0.5) * S;
          if (s === 0) addPaddle(tang, -half, 0, 1, plateW, plateT); // -Z wall
          else if (s === 1) addPaddle(half, tang, -1, 0, plateT, plateW); // +X wall
          else if (s === 2) addPaddle(tang, half, 0, -1, plateW, plateT); // +Z wall
          else addPaddle(-half, tang, 1, 0, plateT, plateW); // -X wall
        }
      }
    } else {
      // Fallback: place each paddle at its exact grid cell.
      for (let k = 0; k < P; k++) {
        const cell = cells[k];
        const ix = cell % n;
        const iy = Math.floor(cell / n);
        const { x, z } = this.gridToWorld(ix, iy);
        const dx = ix === 0 ? 1 : ix === n - 1 ? -1 : 0;
        const dz = iy === 0 ? 1 : iy === n - 1 ? -1 : 0;
        const onXWall = ix === 0 || ix === n - 1;
        addPaddle(x, z, dx, dz, onXWall ? plateT : plateW, onXWall ? plateW : plateT);
      }
    }
  }

  /** Slide each paddle in/out along its wall normal by its current amplitude. */
  setPistonOffsets(amps: ArrayLike<number>): void {
    const s = this.pistonTravelScale;
    const nAmp = amps.length;
    for (let k = 0; k < this.pistons.length; k++) {
      const a = k < nAmp ? amps[k] : 0;
      const base = this.pistonBase[k];
      const dir = this.pistonDir[k];
      this.pistons[k].position.set(base.x + dir.x * a * s, base.y, base.z + dir.z * a * s);
    }
  }

  // ---------------------------------------------------------------------------
  // Caustic on the floor (M-D)
  // ---------------------------------------------------------------------------

  /** Map the live 2D caustic canvas onto the tank floor as a self-lit texture, so
   *  the pattern that forms on the floor matches the target. Refreshed each frame
   *  from `sourceCanvas`. */
  setFloorCaustic(sourceCanvas: HTMLCanvasElement): void {
    const tex = new THREE.CanvasTexture(sourceCanvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    const mat = this.floorMesh.material as THREE.MeshStandardMaterial;
    mat.map = tex;
    mat.emissiveMap = tex;
    mat.emissive = new THREE.Color(0xffffff);
    mat.emissiveIntensity = 1.35; // glow so it reads in the shadowed tank floor
    mat.color = new THREE.Color(0x05080c);
    mat.needsUpdate = true;
    this.causticTex = tex;
  }

  // ---------------------------------------------------------------------------
  // Scene construction
  // ---------------------------------------------------------------------------

  /** Soft image-based environment (a vertical sky→dark gradient) so the water's
   *  reflections reveal the wave shape. */
  private setupEnvironment(): void {
    const c = document.createElement("canvas");
    c.width = 16;
    c.height = 64;
    const ctx = c.getContext("2d")!;
    const g = ctx.createLinearGradient(0, 0, 0, 64);
    g.addColorStop(0.0, "#bcd8ff");
    g.addColorStop(0.45, "#4a6b88");
    g.addColorStop(1.0, "#080c10");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 16, 64);
    const tex = new THREE.CanvasTexture(c);
    tex.mapping = THREE.EquirectangularReflectionMapping;
    tex.colorSpace = THREE.SRGBColorSpace;
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromEquirectangular(tex).texture;
    tex.dispose();
    pmrem.dispose();
  }

  private buildLights(): void {
    const { tankSize: S, tankDepth: D } = this.cfg;

    this.scene.add(new THREE.AmbientLight(0x2a3644, 0.8));
    this.scene.add(new THREE.HemisphereLight(0x9fc4ff, 0x0a0f14, 0.5));

    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(S * 0.25, D * 3.0, S * 0.15);
    key.target.position.set(0, -D, 0);
    this.scene.add(key);
    this.scene.add(key.target);

    const fill = new THREE.DirectionalLight(0x88aaff, 0.4);
    fill.position.set(-S, D * 0.5, S);
    this.scene.add(fill);

    // Glowing disc marking the overhead source.
    const bulb = new THREE.Mesh(
      new THREE.CircleGeometry(S * 0.16, 40),
      new THREE.MeshBasicMaterial({ color: 0xfff3cf, side: THREE.DoubleSide })
    );
    bulb.position.set(0, D * 2.2, 0);
    bulb.rotation.x = -Math.PI / 2;
    this.scene.add(bulb);
  }

  private buildTank(): { water: THREE.Mesh; floor: THREE.Mesh } {
    const { tankSize: S, tankDepth: D, wallThickness: t } = this.cfg;
    const half = S / 2;
    const floorY = -D;
    const rimY = D * 0.12;

    // --- Floor (caustic screen) --------------------------------------------
    const floorGeo = new THREE.PlaneGeometry(S, S);
    floorGeo.rotateX(-Math.PI / 2);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x11202c,
      roughness: 0.95,
      metalness: 0.0,
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.position.y = floorY;
    this.scene.add(floor);

    // --- Glass walls --------------------------------------------------------
    const wallH = rimY - floorY;
    const wallMidY = (rimY + floorY) / 2;
    const glass = new THREE.MeshPhysicalMaterial({
      color: 0xbfe6ff,
      transparent: true,
      opacity: 0.1,
      roughness: 0.05,
      metalness: 0.0,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const wallDefs: [number, number, number, number, number][] = [
      [t, wallH, S + t, +half, 0],
      [t, wallH, S + t, -half, 0],
      [S + t, wallH, t, 0, +half],
      [S + t, wallH, t, 0, -half],
    ];
    for (const [sx, sy, sz, cx, cz] of wallDefs) {
      const wall = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), glass);
      wall.position.set(cx, wallMidY, cz);
      this.scene.add(wall);
    }

    const boxEdges = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(S, wallH, S)),
      new THREE.LineBasicMaterial({ color: 0x4fd8c4, transparent: true, opacity: 0.5 })
    );
    boxEdges.position.y = wallMidY;
    this.scene.add(boxEdges);

    // --- Water surface (textured) ------------------------------------------
    const seg = this.cfg.waterSegments;
    const waterGeo = new THREE.PlaneGeometry(S, S, seg, seg);
    waterGeo.rotateX(-Math.PI / 2); // lie flat: spans X/Z, normal +Y

    this.waterNormalTex = this.makeWaterNormalTexture();
    const waterMat = new THREE.MeshPhysicalMaterial({
      color: 0x2f7fb5,
      transparent: true,
      opacity: 0.72,
      roughness: 0.12,
      metalness: 0.0,
      transmission: 0.4,
      thickness: D,
      ior: 1.333,
      side: THREE.DoubleSide,
      normalMap: this.waterNormalTex,
      normalScale: new THREE.Vector2(0.35, 0.35),
      envMapIntensity: 1.2,
    });
    const water = new THREE.Mesh(waterGeo, waterMat);
    water.position.y = 0;
    this.scene.add(water);

    return { water, floor };
  }

  /** Procedural, seamlessly-tiling ripple normal map (integer wavenumbers ⇒ no
   *  seams). Animated by scrolling its offset each frame for a shimmering micro
   *  surface on top of the macro waves. */
  private makeWaterNormalTexture(size = 256): THREE.CanvasTexture {
    const TWO_PI = Math.PI * 2;
    const height = new Float32Array(size * size);
    const waves: [number, number, number][] = [
      [3, 2, 1.0],
      [5, -3, 0.6],
      [2, 7, 0.5],
      [-6, 4, 0.35],
    ];
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        let h = 0;
        for (const [ax, ay, amp] of waves) {
          h += amp * Math.sin((TWO_PI * (ax * x + ay * y)) / size);
        }
        height[y * size + x] = h;
      }
    }
    const at = (x: number, y: number) =>
      height[((y + size) % size) * size + ((x + size) % size)];

    const c = document.createElement("canvas");
    c.width = c.height = size;
    const ctx = c.getContext("2d")!;
    const img = ctx.createImageData(size, size);
    const strength = 24;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const dhx = (at(x + 1, y) - at(x - 1, y)) * strength;
        const dhy = (at(x, y + 1) - at(x, y - 1)) * strength;
        // tangent-space normal = normalize(-dhx, -dhy, 1)
        const inv = 1 / Math.sqrt(dhx * dhx + dhy * dhy + 1);
        const nx = -dhx * inv;
        const ny = -dhy * inv;
        const nz = inv;
        const i = (y * size + x) * 4;
        img.data[i] = (nx * 0.5 + 0.5) * 255;
        img.data[i + 1] = (ny * 0.5 + 0.5) * 255;
        img.data[i + 2] = (nz * 0.5 + 0.5) * 255;
        img.data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(3, 3);
    return tex;
  }

  // ---------------------------------------------------------------------------
  // Lifecycle
  // ---------------------------------------------------------------------------

  private resize(): void {
    const w = Math.max(1, this.container.clientWidth);
    const h = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** Per-frame visual updates independent of the physics (texture animation). */
  private updateVisuals(dt: number): void {
    this.elapsed += dt;
    if (this.waterNormalTex) {
      this.waterNormalTex.offset.x = (this.waterNormalTex.offset.x + dt * 0.03) % 1;
      this.waterNormalTex.offset.y = (this.waterNormalTex.offset.y + dt * 0.017) % 1;
    }
    if (this.causticTex) this.causticTex.needsUpdate = true; // pull latest floor frame
  }

  /** Render one frame with an externally supplied dt. Use this when a single app
   *  loop drives both the 3D tank and the 2D preview off ONE clock, so they stay
   *  perfectly in sync (see main.ts). */
  frame(dt: number): void {
    this.onFrame?.(dt);
    this.updateVisuals(dt);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  /** Begin a self-contained render loop (standalone use). When the app drives the
   *  frame externally via `frame(dt)`, don't call this. */
  start(): void {
    if (this.running) return;
    this.running = true;
    this.lastTime = performance.now();
    const tick = (now: number) => {
      if (!this.running) return;
      const dt = Math.min(0.05, (now - this.lastTime) / 1000);
      this.lastTime = now;
      this.frame(dt);
      this.rafId = requestAnimationFrame(tick);
    };
    this.rafId = requestAnimationFrame(tick);
  }

  stop(): void {
    this.running = false;
    if (this.rafId) cancelAnimationFrame(this.rafId);
    this.rafId = 0;
  }

  dispose(): void {
    this.stop();
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.renderer.dispose();
  }
}
