import { defineConfig } from 'vite';

// Dev: browser talks only to the Vite origin (:3000). /api is proxied to the
// Kairos API so the session cookie is first-party same-origin. Without this,
// SPA on :3000 → API on :8000 is cross-origin; browsers often drop or refuse
// to re-send the kairos_session cookie and every post-login request 401s,
// bouncing the operator straight back to the sign-in screen.
const API_TARGET = process.env.KAIROS_API_PROXY || 'http://127.0.0.1:8000';

export default defineConfig({
  server: {
    host: '127.0.0.1',
    port: 3000,
    strictPort: false,
    proxy: {
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
      },
    },
  },
  build: {
    chunkSizeWarningLimit: 650,
  },
});
