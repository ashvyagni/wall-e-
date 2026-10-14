/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const apiHttp = process.env.VITE_API_HTTP_URL ?? "http://localhost:8000";
const apiWs = process.env.VITE_API_WS_URL ?? "ws://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": { target: apiHttp, changeOrigin: true },
      "/ws": { target: apiWs, ws: true },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    include: ["tests/**/*.test.{ts,tsx}"],
  },
});
