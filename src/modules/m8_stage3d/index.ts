/**
 * M8 · Stage3D — the interactive 3D tank (presentation layer).
 *
 * This is the "3D interactive environment" milestone track (M-A…M-F). It is a
 * NEW presentation layer on top of the verified 2D pipeline (M1–M7); it does NOT
 * touch the numerics. Three.js (WebGL) renders a glass tank with a water surface,
 * a floor (where the caustic will land), an overhead light, and an orbit camera.
 *
 * Milestone M-A (this file's first cut): a STATIC stage — flat water, no physics.
 *   Acceptance: you can orbit around a static tank with a flat water surface.
 *
 * Later milestones plug in here without a rewrite:
 *   M-B water displacement → mutate `waterGeometry` vertices from the CPU sim.
 *   M-C pistons           → meshes placed via `gridToWorld(ix, iy)`.
 *   M-D caustic on floor  → swap/patch `floorMesh.material.map`.
 * Hence the grid↔world mapping and the public accessors below.
 */

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export interface Stage3DConfig {
  /** Sim grid cells per side (matches the M⁺ asset geometry, e.g. 16). Used by
   *  `gridToWorld` so pistons / water vertices line up with the physics grid. */
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
  waterSegments: 48,
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
  /** Per-water-vertex grid coords (u,v ∈ [0,1]) for bilinear field sampling. */
  private waterU!: Float32Array;
  private waterV!: Float32Array;

  private readonly container: HTMLElement;
  private readonly resizeObserver: ResizeObserver;
  private rafId = 0;
  private running = false;
  /** Optional hook run every frame before render (e.g. drive the sim in M-B). */
  onFrame: ((dtSeconds: number) => void) | null = null;
  private lastTime = 0;

  constructor(canvas: HTMLCanvasElement, cfg: Partial<Stage3DConfig> = {}) {
    this.cfg = { ...DEFAULT_STAGE_CONFIG, ...cfg };
    this.container = (canvas.parentElement ?? canvas) as HTMLElement;

    const { tankSize: S, tankDepth: D } = this.cfg;

    // --- Renderer -----------------------------------------------------------
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setClearColor(0x0b0e11, 1);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // --- Camera + controls --------------------------------------------------
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
    this.camera.position.set(S * 1.15, D * 1.7, S * 1.35);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, -D * 0.35, 0);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = S * 0.6;
    this.controls.maxDistance = S * 6;
    // Full vertical orbit: you can rise directly overhead or dip below to look
    // up through the tank at the light. (No polar clamp.)

    this.buildLights();
    const { water, floor } = this.buildTank();
    this.waterMesh = water;
    this.floorMesh = floor;
    this.precomputeWaterUV();

    // Fit to container now and on every resize.
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    this.resize();
  }

  // ---------------------------------------------------------------------------
  // Grid ↔ world mapping (shared by later milestones)
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

  /** Cache each water vertex's normalised grid coords (u,v) once. The water plane
   *  is subdivided finer than the sim grid, so displacement bilinearly upsamples
   *  the n² field onto these vertices. */
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
  // Scene construction
  // ---------------------------------------------------------------------------

  private buildLights(): void {
    const { tankSize: S, tankDepth: D } = this.cfg;

    this.scene.add(new THREE.AmbientLight(0x2a3644, 0.9));

    const hemi = new THREE.HemisphereLight(0x9fc4ff, 0x0a0f14, 0.5);
    this.scene.add(hemi);

    // Overhead key light — the "light source from above" that casts the caustic.
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(S * 0.25, D * 3.0, S * 0.15);
    key.target.position.set(0, -D, 0);
    this.scene.add(key);
    this.scene.add(key.target);

    // A soft fill from the front so the glass reads.
    const fill = new THREE.DirectionalLight(0x88aaff, 0.4);
    fill.position.set(-S, D * 0.5, S);
    this.scene.add(fill);

    // Visible marker for the overhead source (a glowing disc hovering above).
    const bulb = new THREE.Mesh(
      new THREE.CircleGeometry(S * 0.18, 40),
      new THREE.MeshBasicMaterial({ color: 0xfff3cf, side: THREE.DoubleSide })
    );
    bulb.position.set(0, D * 2.2, 0);
    bulb.rotation.x = -Math.PI / 2;
    this.scene.add(bulb);
  }

  private buildTank(): { water: THREE.Mesh; floor: THREE.Mesh } {
    const { tankSize: S, tankDepth: D, wallThickness: t } = this.cfg;
    const half = S / 2;
    const waterTopY = 0;
    const floorY = -D;
    const rimY = D * 0.12; // glass rises a little above the water line

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

    // Faint grid on the floor for depth cues.
    const grid = new THREE.GridHelper(S, this.cfg.gridN, 0x2b4a5f, 0x1a2e3c);
    (grid.material as THREE.Material).transparent = true;
    (grid.material as THREE.Material).opacity = 0.35;
    grid.position.y = floorY + 0.001;
    this.scene.add(grid);

    // --- Glass walls --------------------------------------------------------
    const wallH = rimY - floorY;
    const wallMidY = (rimY + floorY) / 2;
    const glass = new THREE.MeshPhysicalMaterial({
      color: 0xbfe6ff,
      transparent: true,
      opacity: 0.12,
      roughness: 0.05,
      metalness: 0.0,
      transmission: 0.0, // keep cheap + robust for M-A; simple alpha glass
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const wallDefs: [number, number, number, number, number][] = [
      // [sizeX, sizeY, sizeZ, centerX, centerZ]
      [t, wallH, S + t, +half, 0], // +X
      [t, wallH, S + t, -half, 0], // -X
      [S + t, wallH, t, 0, +half], // +Z
      [S + t, wallH, t, 0, -half], // -Z
    ];
    for (const [sx, sy, sz, cx, cz] of wallDefs) {
      const wall = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), glass);
      wall.position.set(cx, wallMidY, cz);
      this.scene.add(wall);
    }

    // Crisp edge lines around the glass box so its shape reads clearly.
    const boxEdges = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(S, wallH, S)),
      new THREE.LineBasicMaterial({ color: 0x4fd8c4, transparent: true, opacity: 0.5 })
    );
    boxEdges.position.y = wallMidY;
    this.scene.add(boxEdges);

    // --- Water surface ------------------------------------------------------
    const seg = this.cfg.waterSegments;
    const waterGeo = new THREE.PlaneGeometry(S, S, seg, seg);
    waterGeo.rotateX(-Math.PI / 2); // lie flat: spans X/Z, normal +Y
    const waterMat = new THREE.MeshPhysicalMaterial({
      color: 0x2f7fb5,
      transparent: true,
      opacity: 0.55,
      roughness: 0.18,
      metalness: 0.0,
      transmission: 0.35,
      thickness: D,
      side: THREE.DoubleSide,
    });
    const water = new THREE.Mesh(waterGeo, waterMat);
    water.position.y = waterTopY;
    this.scene.add(water);

    return { water, floor };
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

  /** Begin the render loop. */
  start(): void {
    if (this.running) return;
    this.running = true;
    this.lastTime = performance.now();
    const tick = (now: number) => {
      if (!this.running) return;
      const dt = Math.min(0.05, (now - this.lastTime) / 1000);
      this.lastTime = now;
      this.onFrame?.(dt);
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
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
