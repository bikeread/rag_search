// Vitest测试环境配置
import { config } from 'dotenv'
import { afterEach, vi } from 'vitest'

// 加载测试环境变量
config({ path: '.env.test' })

// Mock console methods to avoid noise in tests
global.console = {
  ...console,
  log: vi.fn(),
  debug: vi.fn(),
  info: vi.fn(),
  warn: vi.fn(),
  error: vi.fn(),
}

// Clean up after each test
afterEach(() => {
  vi.clearAllMocks()
})