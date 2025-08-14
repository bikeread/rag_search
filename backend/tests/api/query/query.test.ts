import { createMocks } from 'node-mocks-http'
import { NextApiRequest, NextApiResponse } from 'next'
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { prismaMock } from '../../__mocks__/prisma'
import { CacheService } from '@/lib/redis'
import { ragService } from '@/services/pythonServices'
import handler from '@/pages/api/query/index'
import { AuthenticatedRequest } from '@/lib/jwtAuth'

// Mock dependencies
vi.mock('@/lib/prisma', () => ({
  prisma: prismaMock,
}))

vi.mock('@/lib/redis', () => ({
  CacheService: {
    get: vi.fn(),
    set: vi.fn(),
  },
}))

vi.mock('@/services/pythonServices', () => ({
  ragService: {
    query: vi.fn(),
  },
}))

vi.mock('@/lib/cors', () => ({
  withCorsAndAuth: (handler: any) => handler,
}))

const mockCacheService = CacheService as any
const mockRagService = ragService as any

describe('/api/query', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    prismaMock.query.create.mockResolvedValue({
      id: 'test-query-id',
      text: 'test query',
      userId: 'test-user-id',
      status: 'PROCESSING',
      createdAt: new Date(),
      updatedAt: new Date(),
      response: null,
      responseTime: null,
      sources: null,
      metadata: null,
    })
    prismaMock.query.update.mockResolvedValue({
      id: 'test-query-id',
      text: 'test query',
      userId: 'test-user-id',
      status: 'COMPLETED',
      response: 'test response',
      responseTime: 1000,
      sources: [],
      metadata: {},
      createdAt: new Date(),
      updatedAt: new Date(),
    })
  })

  describe('POST /api/query', () => {
    it('应该成功处理查询请求并返回结果', async () => {
      // Mock RAG service response
      mockRagService.query.mockResolvedValue({
        answer: 'RAG是检索增强生成技术',
        sources: [
          {
            id: 'doc1',
            score: 0.9,
            text: 'RAG相关内容',
            metadata: { document_id: 'doc1' }
          }
        ],
        query_time: 500,
      })

      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'POST',
        body: {
          query: '什么是RAG？',
          topK: 3,
          useCache: false,
        },
      })

      // Mock authenticated user
      req.user = {
        id: 'test-user-id',
        email: 'test@example.com',
        name: 'Test User',
        role: 'USER',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(200)
      const data = JSON.parse(res._getData())
      expect(data).toMatchObject({
        queryId: 'test-query-id',
        answer: 'RAG是检索增强生成技术',
        sources: expect.any(Array),
        responseTime: expect.any(Number),
        metadata: expect.any(Object),
      })

      // Verify database interactions
      expect(prismaMock.query.create).toHaveBeenCalledWith({
        data: {
          text: '什么是RAG？',
          userId: 'test-user-id',
          status: 'PROCESSING',
        },
      })

      expect(prismaMock.query.update).toHaveBeenCalledWith({
        where: { id: 'test-query-id' },
        data: {
          response: 'RAG是检索增强生成技术',
          responseTime: expect.any(Number),
          sources: expect.any(Array),
          status: 'COMPLETED',
          metadata: expect.any(Object),
        },
      })

      // Verify RAG service was called
      expect(mockRagService.query).toHaveBeenCalledWith('什么是RAG？', 3, true)
    })

    it('应该验证请求参数并拒绝无效请求', async () => {
      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'POST',
        body: {
          query: '', // 空查询
          topK: 25,  // 超出限制
        },
      })

      req.user = {
        id: 'test-user-id',
        email: 'test@example.com',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(400)
      const data = JSON.parse(res._getData())
      expect(data.error).toBe('Invalid query data')
      expect(data.details).toBeDefined()
    })

    it('应该处理缓存的查询结果', async () => {
      const cachedResult = {
        queryId: 'cached-query-id',
        answer: 'Cached answer',
        sources: [],
        responseTime: 100,
        metadata: { cached: true },
      }

      mockCacheService.get.mockResolvedValue(cachedResult)

      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'POST',
        body: {
          query: 'cached query',
          topK: 3,
          useCache: true,
        },
      })

      req.user = {
        id: 'test-user-id',
        email: 'test@example.com',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(200)
      const data = JSON.parse(res._getData())
      expect(data.cached).toBe(true)
      expect(data.answer).toBe('Cached answer')

      // Verify cache was checked
      const expectedCacheKey = `query:${Buffer.from('cached query').toString('base64')}:3`
      expect(mockCacheService.get).toHaveBeenCalledWith(expectedCacheKey)

      // Verify RAG service was not called
      expect(mockRagService.query).not.toHaveBeenCalled()
    })

    it('应该处理RAG服务错误', async () => {
      mockRagService.query.mockRejectedValue(new Error('RAG service error'))

      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'POST',
        body: {
          query: '测试查询',
          topK: 3,
        },
      })

      req.user = {
        id: 'test-user-id',
        email: 'test@example.com',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(500)
      const data = JSON.parse(res._getData())
      expect(data.error).toBe('Query failed')

      // Verify query record was updated to failed status
      expect(prismaMock.query.update).toHaveBeenCalledWith({
        where: { id: 'test-query-id' },
        data: {
          status: 'FAILED',
          responseTime: expect.any(Number),
          metadata: {
            error: 'RAG service error',
          },
        },
      })
    })

    it('应该只接受POST请求', async () => {
      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'GET',
      })

      req.user = {
        id: 'test-user-id',
        email: 'test@example.com',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(405)
      const data = JSON.parse(res._getData())
      expect(data.error).toBe('Method not allowed')
    })

    it('应该正确设置缓存键', async () => {
      mockRagService.query.mockResolvedValue({
        answer: 'test answer',
        sources: [],
        query_time: 100,
      })

      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'POST',
        body: {
          query: 'test query with 中文',
          topK: 5,
          useCache: true,
        },
      })

      req.user = {
        id: 'test-user-id',
        email: 'test@example.com',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(200)

      // Verify cache set was called with correct key
      const expectedCacheKey = `query:${Buffer.from('test query with 中文').toString('base64')}:5`
      expect(mockCacheService.set).toHaveBeenCalledWith(
        expectedCacheKey,
        expect.any(Object),
        3600 // 1 hour TTL
      )
    })

    it('应该处理缺少sources的RAG响应', async () => {
      mockRagService.query.mockResolvedValue({
        answer: 'answer without sources',
        query_time: 200,
        // sources 字段缺失
      })

      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'POST',
        body: {
          query: 'query without sources',
          topK: 3,
        },
      })

      req.user = {
        id: 'test-user-id',
        email: 'test@example.com',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(200)
      const data = JSON.parse(res._getData())
      expect(data.sources).toEqual([])
      expect(data.metadata.sourcesCount).toBe(0)
    })

    it('应该正确处理长查询文本', async () => {
      const longQuery = 'a'.repeat(1000) // 最大长度查询

      mockRagService.query.mockResolvedValue({
        answer: 'response to long query',
        sources: [],
        query_time: 300,
      })

      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'POST',
        body: {
          query: longQuery,
          topK: 3,
        },
      })

      req.user = {
        id: 'test-user-id',
        email: 'test@example.com',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(200)
      expect(mockRagService.query).toHaveBeenCalledWith(longQuery, 3, true)
    })

    it('应该拒绝超长查询文本', async () => {
      const tooLongQuery = 'a'.repeat(1001) // 超过最大长度

      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'POST',
        body: {
          query: tooLongQuery,
          topK: 3,
        },
      })

      req.user = {
        id: 'test-user-id',
        email: 'test@example.com',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(400)
      expect(mockRagService.query).not.toHaveBeenCalled()
    })
  })
})