import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";
import fs from "fs";

const version = fs.readFileSync(path.resolve(__dirname, "../VERSION"), "utf-8").trim();

export default defineConfig({
  plugins: [react(), tailwindcss()],
  define: {
    __APP_VERSION__: JSON.stringify(version),
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  optimizeDeps: {
    // QuickJS imports its Emscripten WASM through package-relative URLs.
    // Prebundling rewrites those URLs into /node_modules/.vite/deps without
    // copying the sibling WASM asset.
    exclude: ["quickjs-emscripten"],
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
  },
  build: {
    chunkSizeWarningLimit: 750,
    rollupOptions: {
      output: {
        manualChunks: {
          three: ["three"],
          "three-ecosystem": [
            "@react-three/fiber",
            "@react-three/drei",
            "@react-three/postprocessing",
          ],
          recharts: ["recharts"],
          react: ["react", "react-dom", "react-router-dom"],
        },
      },
    },
  },
});
