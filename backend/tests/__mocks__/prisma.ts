import { vi, beforeEach } from 'vitest'
import type { PrismaClient } from '@prisma/client'

// Create a mock Prisma client
const prismaMock = {
  query: {
    create: vi.fn(),
    update: vi.fn(),
    findMany: vi.fn(),
    count: vi.fn(),
  },
  chatSession: {
    create: vi.fn(),
  },
  chatMessage: {
    create: vi.fn(),
  },
  document: {
    create: vi.fn(),
    findMany: vi.fn(),
    findUnique: vi.fn(),
    update: vi.fn(),
    delete: vi.fn(),
  },
  user: {
    create: vi.fn(),
    findUnique: vi.fn(),
  },
} as any

beforeEach(() => {
  vi.clearAllMocks()
})

export { prismaMock }