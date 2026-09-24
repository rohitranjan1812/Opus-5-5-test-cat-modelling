import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The dev server proxies the API; `npm run build` emits dist/ which FastAPI serves at /.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
      '/docs': 'http://localhost:8000',
      '/openapi.json': 'http://localhost:8000',
    },
  },
  build: { outDir: 'dist', chunkSizeWarningLimit: 2000 },
})
