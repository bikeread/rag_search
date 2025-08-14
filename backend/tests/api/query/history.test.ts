import { createMocks } from 'node-mocks-http'
import { NextApiRequest, NextApiResponse } from 'next'
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { prismaMock } from '../../__mocks__/prisma'
import handler from '@/pages/api/query/history'
import { AuthenticatedRequest } from '@/lib/jwtAuth'

// Mock dependencies
vi.mock('@/lib/prisma', () => ({
  prisma: prismaMock,
}))

vi.mock('@/lib/cors', () => ({
  withCorsAndAuth: (handler: any) => handler,
}))

describe('/api/query/history', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('GET /api/query/history', () => {
    it('应该返回用户的查询历史记录', async () => {
      const mockQueries = [
        {
          id: 'query-1',
          text: '什么是RAG？',
          response: 'RAG是检索增强生成技术',
          status: 'COMPLETED',
          responseTime: 1000,
          sources: [{ id: 'doc1', score: 0.9 }],
          metadata: { topK: 5 },
          createdAt: new Date('2024-01-01T10:00:00Z'),
          updatedAt: new Date('2024-01-01T10:00:01Z'),
        },
        {
          id: 'query-2',
          text: '如何使用向量数据库？',
          response: '向量数据库是专门存储向量的数据库',
          status: 'COMPLETED',
          responseTime: 1500,
          sources: [{ id: 'doc2', score: 0.8 }],
          metadata: { topK: 3 },
          createdAt: new Date('2024-01-01T11:00:00Z'),
          updatedAt: new Date('2024-01-01T11:00:02Z'),
        },
      ]

      prismaMock.query.findMany.mockResolvedValue(mockQueries)
      prismaMock.query.count.mockResolvedValue(2)

      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'GET',
        query: {
          limit: '10',
          page: '1',
        },
      })

      req.user = {
        id: 'test-user-id',
        email: 'test@example.com',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(200)
      const data = JSON.parse(res._getData())
      
      expect(data.queries).toHaveLength(2)
      expect(data.pagination.total).toBe(2)
      expect(data.queries[0]).toMatchObject({
        id: 'query-1',
        text: '什么是RAG？',
        response: 'RAG是检索增强生成技术',
        status: 'COMPLETED',
        responseTime: 1000,
      })

      // Verify database query
      expect(prismaMock.query.findMany).toHaveBeenCalledWith({
        where: {
          userId: 'test-user-id',
          status: {
            not: 'DELETED'
          }
        },
        orderBy: {
          createdAt: 'desc',
        },
        take: 10,
        skip: 0,
        select: {
          id: true,
          text: true,
          response: true,
          responseTime: true,
          status: true,
          metadata: true,
          createdAt: true,
        },
      })
    })

    it('应该支持分页参数', async () => {
      prismaMock.query.findMany.mockResolvedValue([])
      prismaMock.query.count.mockResolvedValue(50)

      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'GET',
        query: {
          limit: '5',
          page: '3',  // page 3 with limit 5 = skip 10
        },
      })

      req.user = {
        id: 'test-user-id',
        email: 'test@example.com',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(200)

      expect(prismaMock.query.findMany).toHaveBeenCalledWith({
        where: {
          userId: 'test-user-id',
          status: {
            not: 'DELETED'
          }
        },
        orderBy: {
          createdAt: 'desc',
        },
        take: 5,
        skip: 10,
        select: expect.any(Object),
      })
    })

    it('应该使用默认分页参数', async () => {
      prismaMock.query.findMany.mockResolvedValue([])
      prismaMock.query.count.mockResolvedValue(0)

      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'GET',
        // 没有查询参数
      })

      req.user = {
        id: 'test-user-id',
        email: 'test@example.com',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(200)

      // 验证使用了默认参数
      expect(prismaMock.query.findMany).toHaveBeenCalledWith({
        where: {
          userId: 'test-user-id',
          status: {
            not: 'DELETED'
          }
        },
        orderBy: {
          createdAt: 'desc',
        },
        take: 20,  // 默认limit
        skip: 0,   // 默认offset
        select: expect.any(Object),
      })
    })

    it('应该限制最大查询数量', async () => {
      prismaMock.query.findMany.mockResolvedValue([])
      prismaMock.query.count.mockResolvedValue(0)

      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'GET',
        query: {
          limit: '200', // 超过最大限制
        },
      })

      req.user = {
        id: 'test-user-id',
        email: 'test@example.com',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(200)

      // 验证限制为100
      expect(prismaMock.query.findMany).toHaveBeenCalledWith({
        where: {
          userId: 'test-user-id',
          status: {
            not: 'DELETED'
          }
        },
        orderBy: {
          createdAt: 'desc',
        },
        take: 100,  // 被限制为最大值
        skip: 0,
        select: expect.any(Object),
      })
    })

    it('应该处理无效的分页参数', async () => {
      prismaMock.query.findMany.mockResolvedValue([])
      prismaMock.query.count.mockResolvedValue(0)

      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'GET',
        query: {
          limit: 'invalid',
          offset: 'also-invalid',
        },
      })

      req.user = {
        id: 'test-user-id',
        email: 'test@example.com',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(200)

      // 验证使用默认参数
      expect(prismaMock.query.findMany).toHaveBeenCalledWith({
        where: {
          userId: 'test-user-id',
          status: {
            not: 'DELETED'
          }
        },
        orderBy: {
          createdAt: 'desc',
        },
        take: 20,
        skip: 0,
        select: expect.any(Object),
      })
    })

    it('应该处理数据库错误', async () => {
      prismaMock.query.findMany.mockRejectedValue(new Error('Database error'))

      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'GET',
      })

      req.user = {
        id: 'test-user-id',
        email: 'test@example.com',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(500)
      const data = JSON.parse(res._getData())
      expect(data.error).toBe('Failed to get query history')
    })

    it('应该只返回当前用户的查询记录', async () => {
      const mockQueries = [
        {
          id: 'query-1',
          text: 'User query',
          response: 'Response',
          status: 'COMPLETED',
          responseTime: 1000,
          sources: [],
          metadata: {},
          createdAt: new Date(),
          updatedAt: new Date(),
        },
      ]

      prismaMock.query.findMany.mockResolvedValue(mockQueries)
      prismaMock.query.count.mockResolvedValue(1)

      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'GET',
      })

      req.user = {
        id: 'specific-user-id',
        email: 'specific@example.com',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(200)

      // 验证只查询特定用户的记录
      expect(prismaMock.query.findMany).toHaveBeenCalledWith({
        where: {
          userId: 'specific-user-id',
          status: {
            not: 'DELETED'
          }
        },
        orderBy: {
          createdAt: 'desc',
        },
        take: 20,
        skip: 0,
        select: expect.any(Object),
      })
    })

    it('应该只接受GET请求', async () => {
      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'POST',
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

    it('应该返回空结果当用户没有查询历史时', async () => {
      prismaMock.query.findMany.mockResolvedValue([])
      prismaMock.query.count.mockResolvedValue(0)

      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'GET',
      })

      req.user = {
        id: 'new-user-id',
        email: 'new@example.com',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(200)
      const data = JSON.parse(res._getData())
      
      expect(data.queries).toEqual([])
      expect(data.pagination.total).toBe(0)
    })
  })
})