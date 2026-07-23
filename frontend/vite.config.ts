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
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
  },
  build: {
    chunkSizeWarningLimit: 750,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("node_modules/three/")) return "three";
          if (
            id.includes("node_modules/@react-three/fiber/") ||
            id.includes("node_modules/@react-three/drei/") ||
            id.includes("node_modules/@react-three/postprocessing/")
          ) {
            return "three-ecosystem";
          }
          if (id.includes("node_modules/recharts/")) return "recharts";
          if (
            id.includes("node_modules/react/") ||
            id.includes("node_modules/react-dom/") ||
            id.includes("node_modules/react-router-dom/")
          ) {
            return "react";
          }
        },
      },
    },
  },
});
