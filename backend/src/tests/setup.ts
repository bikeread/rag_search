import { beforeAll, afterAll, vi } from 'vitest'

// Mock系统核心模块
vi.mock('@/lib/cors', () => ({
  withCorsAndAuth: (handler: Function) => handler,
}))

vi.mock('@/lib/jwtAuth', () => ({
  verifyToken: vi.fn().mockResolvedValue({
    id: 'user-123',
    email: 'test@example.com',
  }),
}))

vi.mock('@/lib/middleware', () => ({
  withParamValidation: (schema: any, handler: Function) => handler,
  globalErrorHandler: (handler: Function) => handler,
}))

vi.mock('@/lib/prisma', () => ({
  prisma: {
    document: {
      findMany: vi.fn(),
      count: vi.fn(),
      create: vi.fn(),
      update: vi.fn(),
      findFirst: vi.fn(),
      findUnique: vi.fn(),
    },
    $transaction: vi.fn(),
    $disconnect: vi.fn(),
  },
}))

vi.mock('@/lib/redis', () => ({
  CacheService: {
    get: vi.fn(),
    set: vi.fn(),
    del: vi.fn(),
    keys: vi.fn(),
    clear: vi.fn(),
  },
}))

vi.mock('@/lib/transaction', () => ({
  withTransaction: vi.fn().mockImplementation((callback) => 
    callback({
      document: {
        create: vi.fn(),
        update: vi.fn(),
        findMany: vi.fn(),
        count: vi.fn(),
        findFirst: vi.fn(),
        findUnique: vi.fn(),
      }
    })
  ),
}))

vi.mock('@/services/pythonServices', () => ({
  documentProcessor: {
    uploadDocument: vi.fn(),
    getStatus: vi.fn(),
    deleteDocument: vi.fn(),
  },
}))

vi.mock('formidable', () => ({
  default: vi.fn().mockImplementation(() => ({
    parse: vi.fn(),
  })),
}))

vi.mock('fs/promises', () => ({
  default: {
    readFile: vi.fn(),
    writeFile: vi.fn(),
    unlink: vi.fn(),
    mkdir: vi.fn(),
  },
  readFile: vi.fn(),
  writeFile: vi.fn(),
  unlink: vi.fn(),
  mkdir: vi.fn(),
}))

beforeAll(async () => {
  process.env.NODE_ENV = 'test'
  process.env.DATABASE_URL = process.env.TEST_DATABASE_URL || process.env.DATABASE_URL
  process.env.JWT_SECRET = 'test-jwt-secret'
  process.env.REDIS_URL = process.env.TEST_REDIS_URL || process.env.REDIS_URL
})

afterAll(async () => {
  // No need to disconnect as prisma is mocked
})