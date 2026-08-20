import { defineConfig } from "vite";

// WGSL shaders are imported as raw strings via `?raw`.
//
// `base` must match the GitHub Pages project path so built asset URLs resolve at
// https://<user>.github.io/Dynamic-Fluid-Caustics/. Only applied for the production
// build; the dev server stays at "/".
export default defineConfig(({ command }) => ({
  base: command === "build" ? "/Dynamic-Fluid-Caustics/" : "/",
  server: { port: 5173, open: true },
  build: { target: "esnext" },
}));
