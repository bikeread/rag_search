import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { createMocks } from 'node-mocks-http'
import handler from '@/pages/api/documents/status/[id]'
import { prisma } from '@/lib/prisma'
import { documentProcessor } from '@/services/pythonServices'
import type { AuthenticatedRequest } from '@/lib/jwtAuth'

vi.mock('@/lib/prisma', () => ({
  prisma: {
    document: {
      findFirst: vi.fn(),
    },
  },
}))

vi.mock('@/services/pythonServices', () => ({
  documentProcessor: {
    getProcessingStatus: vi.fn(),
  },
}))

describe('/api/documents/status/[id]', () => {
  const mockUser = {
    id: 'user-123',
    email: 'test@example.com',
  }

  const mockDocument = {
    id: 'doc-123',
    filename: 'test.pdf',
    originalName: '测试文档.pdf',
    mimeType: 'application/pdf',
    size: 1048576,
    status: 'COMPLETED',
    errorMessage: null,
    uploadedBy: 'user-123',
    createdAt: new Date('2025-08-10T14:00:00.000Z'),
    updatedAt: new Date('2025-08-10T14:05:00.000Z'),
    processingStartedAt: new Date('2025-08-10T14:00:00.000Z'),
    processingCompletedAt: new Date('2025-08-10T14:05:00.000Z'),
    chunks: [
      {
        id: 'chunk-1',
        chunkIndex: 0,
        createdAt: new Date('2025-08-10T14:01:00.000Z'),
      },
      {
        id: 'chunk-2',
        chunkIndex: 1,
        createdAt: new Date('2025-08-10T14:02:00.000Z'),
      },
      {
        id: 'chunk-3',
        chunkIndex: 2,
        createdAt: new Date('2025-08-10T14:03:00.000Z'),
      },
    ],
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.resetAllMocks()
  })

  describe('文档状态查询测试', () => {
    it('应该返回文档的完整状态信息', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'GET',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockResolvedValue(mockDocument)

      await handler(req, res)

      expect(res._getStatusCode()).toBe(200)
      const responseData = JSON.parse(res._getData())

      expect(responseData).toEqual({
        document: {
          id: 'doc-123',
          filename: 'test.pdf',
          originalName: '测试文档.pdf',
          mimeType: 'application/pdf',
          size: 1048576,
          status: 'COMPLETED',
          errorMessage: null,
          chunksCount: 3,
          createdAt: mockDocument.createdAt.toISOString(),
          updatedAt: mockDocument.updatedAt.toISOString(),
          processingStartedAt: mockDocument.processingStartedAt?.toISOString(),
          processingCompletedAt: mockDocument.processingCompletedAt?.toISOString(),
        },
        processingStatus: null,
        chunks: [
          {
            id: 'chunk-1',
            chunkIndex: 0,
            createdAt: mockDocument.chunks[0].createdAt.toISOString(),
          },
          {
            id: 'chunk-2',
            chunkIndex: 1,
            createdAt: mockDocument.chunks[1].createdAt.toISOString(),
          },
          {
            id: 'chunk-3',
            chunkIndex: 2,
            createdAt: mockDocument.chunks[2].createdAt.toISOString(),
          },
        ],
      })
    })

    it('应该拒绝非GET请求', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'POST',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      await handler(req, res)

      expect(res._getStatusCode()).toBe(405)
      expect(JSON.parse(res._getData())).toEqual({
        error: 'Method not allowed',
      })
    })

    it('应该验证文档ID参数', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'GET',
        query: {},
        user: mockUser,
      })

      await handler(req, res)

      expect(res._getStatusCode()).toBe(400)
      expect(JSON.parse(res._getData())).toEqual({
        error: 'Invalid document ID',
      })
    })

    it('应该处理文档ID数组的情况', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'GET',
        query: { id: ['doc-1', 'doc-2'] },
        user: mockUser,
      })

      await handler(req, res)

      expect(res._getStatusCode()).toBe(400)
      expect(JSON.parse(res._getData())).toEqual({
        error: 'Invalid document ID',
      })
    })
  })

  describe('权限验证测试', () => {
    it('应该只能查询当前用户的文档', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'GET',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockResolvedValue(mockDocument)

      await handler(req, res)

      expect(prisma.document.findFirst).toHaveBeenCalledWith({
        where: {
          id: 'doc-123',
          uploadedBy: 'user-123',
        },
        include: {
          chunks: {
            select: {
              id: true,
              chunkIndex: true,
              createdAt: true,
            },
            orderBy: {
              chunkIndex: 'asc',
            },
          },
        },
      })
    })

    it('应该阻止查询其他用户的文档', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'GET',
        query: { id: 'doc-456' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockResolvedValue(null)

      await handler(req, res)

      expect(res._getStatusCode()).toBe(404)
      expect(JSON.parse(res._getData())).toEqual({
        error: 'Document not found',
      })
    })
  })

  describe('处理中文档状态测试', () => {
    it('应该获取处理中文档的Python服务状态', async () => {
      const processingDocument = {
        ...mockDocument,
        status: 'PROCESSING',
        processingCompletedAt: null,
        chunks: [],
      }

      const pythonStatus = {
        progress: 60,
        currentStep: 'Extracting text',
        estimatedTime: 120,
        processedChunks: 15,
        totalChunks: 25,
      }

      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'GET',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockResolvedValue(processingDocument)
      vi.mocked(documentProcessor.getProcessingStatus).mockResolvedValue(pythonStatus)

      await handler(req, res)

      expect(res._getStatusCode()).toBe(200)
      const responseData = JSON.parse(res._getData())

      expect(responseData.document.status).toBe('PROCESSING')
      expect(responseData.processingStatus).toEqual(pythonStatus)

      expect(documentProcessor.getProcessingStatus).toHaveBeenCalledWith('doc-123')
    })

    it('应该处理Python服务状态查询失败', async () => {
      const processingDocument = {
        ...mockDocument,
        status: 'PROCESSING',
        processingCompletedAt: null,
      }

      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'GET',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockResolvedValue(processingDocument)
      vi.mocked(documentProcessor.getProcessingStatus).mockRejectedValue(
        new Error('Python service unavailable')
      )

      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation()

      await handler(req, res)

      expect(res._getStatusCode()).toBe(200)
      const responseData = JSON.parse(res._getData())

      expect(responseData.document.status).toBe('PROCESSING')
      expect(responseData.processingStatus).toBeNull()

      expect(consoleWarnSpy).toHaveBeenCalledWith(
        'Failed to get processing status from Python service:',
        expect.any(Error)
      )

      consoleWarnSpy.mockRestore()
    })

    it('不应该为已完成的文档查询Python服务状态', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'GET',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockResolvedValue(mockDocument)

      await handler(req, res)

      expect(documentProcessor.getProcessingStatus).not.toHaveBeenCalled()
    })
  })

  describe('文档块数据测试', () => {
    it('应该按chunkIndex排序返回文档块', async () => {
      const unorderedChunks = [
        { id: 'chunk-3', chunkIndex: 2, createdAt: new Date() },
        { id: 'chunk-1', chunkIndex: 0, createdAt: new Date() },
        { id: 'chunk-2', chunkIndex: 1, createdAt: new Date() },
      ]

      const documentWithUnorderedChunks = {
        ...mockDocument,
        chunks: unorderedChunks,
      }

      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'GET',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockResolvedValue(documentWithUnorderedChunks)

      await handler(req, res)

      expect(prisma.document.findFirst).toHaveBeenCalledWith(
        expect.objectContaining({
          include: {
            chunks: expect.objectContaining({
              orderBy: {
                chunkIndex: 'asc',
              },
            }),
          },
        })
      )
    })

    it('应该只返回必要的文档块字段', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'GET',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockResolvedValue(mockDocument)

      await handler(req, res)

      expect(prisma.document.findFirst).toHaveBeenCalledWith(
        expect.objectContaining({
          include: {
            chunks: {
              select: {
                id: true,
                chunkIndex: true,
                createdAt: true,
              },
              orderBy: expect.any(Object),
            },
          },
        })
      )
    })

    it('应该正确统计文档块数量', async () => {
      const documentWithManyChunks = {
        ...mockDocument,
        chunks: Array.from({ length: 50 }, (_, i) => ({
          id: `chunk-${i}`,
          chunkIndex: i,
          createdAt: new Date(),
        })),
      }

      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'GET',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockResolvedValue(documentWithManyChunks)

      await handler(req, res)

      const responseData = JSON.parse(res._getData())
      expect(responseData.document.chunksCount).toBe(50)
      expect(responseData.chunks).toHaveLength(50)
    })
  })

  describe('失败状态文档测试', () => {
    it('应该返回失败文档的错误信息', async () => {
      const failedDocument = {
        ...mockDocument,
        status: 'FAILED',
        errorMessage: 'Unsupported file format',
        processingCompletedAt: null,
        chunks: [],
      }

      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'GET',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockResolvedValue(failedDocument)

      await handler(req, res)

      const responseData = JSON.parse(res._getData())
      expect(responseData.document.status).toBe('FAILED')
      expect(responseData.document.errorMessage).toBe('Unsupported file format')
      expect(responseData.document.chunksCount).toBe(0)
      expect(responseData.processingStatus).toBeNull()
    })
  })

  describe('错误处理测试', () => {
    it('应该处理数据库查询错误', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'GET',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockRejectedValue(
        new Error('Database connection error')
      )

      await handler(req, res)

      expect(res._getStatusCode()).toBe(500)
      expect(JSON.parse(res._getData())).toEqual({
        error: 'Failed to get document status',
      })
    })

    it('应该记录错误日志', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'GET',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      const error = new Error('Test error')
      vi.mocked(prisma.document.findFirst).mockRejectedValue(error)

      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation()

      await handler(req, res)

      expect(consoleErrorSpy).toHaveBeenCalledWith('Status check error:', error)

      consoleErrorSpy.mockRestore()
    })
  })

  describe('特殊场景测试', () => {
    it('应该处理没有文档块的文档', async () => {
      const documentWithoutChunks = {
        ...mockDocument,
        chunks: [],
      }

      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'GET',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockResolvedValue(documentWithoutChunks)

      await handler(req, res)

      const responseData = JSON.parse(res._getData())
      expect(responseData.document.chunksCount).toBe(0)
      expect(responseData.chunks).toEqual([])
    })

    it('应该处理处理时间戳缺失的情况', async () => {
      const documentWithoutTimestamps = {
        ...mockDocument,
        processingStartedAt: null,
        processingCompletedAt: null,
      }

      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'GET',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockResolvedValue(documentWithoutTimestamps)

      await handler(req, res)

      const responseData = JSON.parse(res._getData())
      expect(responseData.document.processingStartedAt).toBeUndefined()
      expect(responseData.document.processingCompletedAt).toBeUndefined()
    })

    it('应该处理已删除的文档状态查询', async () => {
      const deletedDocument = {
        ...mockDocument,
        status: 'DELETED',
      }

      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'GET',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockResolvedValue(deletedDocument)

      await handler(req, res)

      const responseData = JSON.parse(res._getData())
      expect(responseData.document.status).toBe('DELETED')
    })
  })

  describe('并发查询测试', () => {
    it('应该正确处理并发状态查询', async () => {
      const documents = Array.from({ length: 5 }, (_, i) => ({
        ...mockDocument,
        id: `doc-${i}`,
      }))

      const promises = documents.map(async (doc) => {
        const { req, res } = createMocks<AuthenticatedRequest>({
          method: 'GET',
          query: { id: doc.id },
          user: mockUser,
        })

        vi.mocked(prisma.document.findFirst).mockResolvedValueOnce(doc)

        await handler(req, res)

        return {
          status: res._getStatusCode(),
          data: JSON.parse(res._getData()),
        }
      })

      const results = await Promise.all(promises)

      results.forEach((result, index) => {
        expect(result.status).toBe(200)
        expect(result.data.document.id).toBe(`doc-${index}`)
      })
    })
  })
})