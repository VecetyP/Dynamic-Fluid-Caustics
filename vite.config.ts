import { defineConfig } from "vite";

// WGSL shaders are imported as raw strings via `?raw`.
export default defineConfig({
  server: { port: 5173, open: true },
  build: { target: "esnext" },
});
