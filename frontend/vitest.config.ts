import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/tests/setup.ts'],
    include: [
      'src/tests/**/*.test.ts',
      'src/tests/**/*.test.tsx',
      'src/tests/**/*.spec.ts',
      'src/tests/**/*.spec.tsx',
    ],
    exclude: [
      'node_modules/**',
      'dist/**',
      'build/**',
      'src/tests/e2e/**', // 排除E2E测试，避免Playwright冲突
    ],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/**',
        'src/tests/**',
        'dist/**',
        'build/**',
        'src/vite-env.d.ts',
      ],
    },
    testTimeout: 10000,
    hookTimeout: 10000,
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
})