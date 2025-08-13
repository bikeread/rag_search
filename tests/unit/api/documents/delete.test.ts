import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { createMocks } from 'node-mocks-http'
import handler from '@/pages/api/documents/delete/[id]'
import { prisma } from '@/lib/prisma'
import { CacheService } from '@/lib/redis'
import type { AuthenticatedRequest } from '@/lib/jwtAuth'

vi.mock('@/lib/prisma', () => ({
  prisma: {
    document: {
      findFirst: vi.fn(),
      update: vi.fn(),
    },
  },
}))

vi.mock('@/lib/redis', () => ({
  CacheService: {
    del: vi.fn(),
  },
}))

describe('/api/documents/delete/[id]', () => {
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
    uploadedBy: 'user-123',
    createdAt: new Date(),
    updatedAt: new Date(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.resetAllMocks()
  })

  describe('文档删除功能测试', () => {
    it('应该成功删除用户自己的文档', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'DELETE',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockResolvedValue(mockDocument)
      vi.mocked(prisma.document.update).mockResolvedValue({
        ...mockDocument,
        status: 'DELETED',
        updatedAt: new Date(),
      })
      vi.mocked(CacheService.del).mockResolvedValue(1)

      await handler(req, res)

      expect(res._getStatusCode()).toBe(200)
      expect(JSON.parse(res._getData())).toEqual({
        message: 'Document deleted successfully',
        documentId: 'doc-123',
      })

      expect(prisma.document.findFirst).toHaveBeenCalledWith({
        where: {
          id: 'doc-123',
          uploadedBy: 'user-123',
        },
      })

      expect(prisma.document.update).toHaveBeenCalledWith({
        where: { id: 'doc-123' },
        data: {
          status: 'DELETED',
          updatedAt: expect.any(Date),
        },
      })

      expect(CacheService.del).toHaveBeenCalledWith('documents:user-123')
    })

    it('应该拒绝非DELETE请求', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'GET',
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
        method: 'DELETE',
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
        method: 'DELETE',
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
    it('应该阻止用户删除其他用户的文档', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'DELETE',
        query: { id: 'doc-456' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockResolvedValue(null)

      await handler(req, res)

      expect(res._getStatusCode()).toBe(404)
      expect(JSON.parse(res._getData())).toEqual({
        error: 'Document not found',
      })

      expect(prisma.document.findFirst).toHaveBeenCalledWith({
        where: {
          id: 'doc-456',
          uploadedBy: 'user-123',
        },
      })

      expect(prisma.document.update).not.toHaveBeenCalled()
    })

    it('应该只查询当前用户的文档', async () => {
      const otherUserDocument = {
        ...mockDocument,
        uploadedBy: 'user-456',
      }

      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'DELETE',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockResolvedValue(null)

      await handler(req, res)

      expect(res._getStatusCode()).toBe(404)
      expect(prisma.document.update).not.toHaveBeenCalled()
    })
  })

  describe('软删除测试', () => {
    it('应该将文档状态标记为DELETED而不是物理删除', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'DELETE',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockResolvedValue(mockDocument)
      vi.mocked(prisma.document.update).mockResolvedValue({
        ...mockDocument,
        status: 'DELETED',
        updatedAt: new Date(),
      })

      await handler(req, res)

      expect(prisma.document.update).toHaveBeenCalledWith({
        where: { id: 'doc-123' },
        data: {
          status: 'DELETED',
          updatedAt: expect.any(Date),
        },
      })

      expect(prisma.document.update).not.toHaveBeenCalledWith(
        expect.objectContaining({
          delete: expect.anything(),
        })
      )
    })

    it('应该更新文档的updatedAt时间戳', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'DELETE',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockResolvedValue(mockDocument)
      
      const beforeDelete = new Date()
      vi.mocked(prisma.document.update).mockImplementation(async ({ data }) => {
        const updatedAt = data.updatedAt as Date
        expect(updatedAt.getTime()).toBeGreaterThanOrEqual(beforeDelete.getTime())
        return {
          ...mockDocument,
          status: 'DELETED',
          updatedAt,
        }
      })

      await handler(req, res)

      expect(prisma.document.update).toHaveBeenCalled()
    })
  })

  describe('缓存清理测试', () => {
    it('应该清除用户的文档缓存', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'DELETE',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockResolvedValue(mockDocument)
      vi.mocked(prisma.document.update).mockResolvedValue({
        ...mockDocument,
        status: 'DELETED',
      })
      vi.mocked(CacheService.del).mockResolvedValue(1)

      await handler(req, res)

      expect(CacheService.del).toHaveBeenCalledWith('documents:user-123')
    })

    it('应该处理缓存清理失败的情况', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'DELETE',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockResolvedValue(mockDocument)
      vi.mocked(prisma.document.update).mockResolvedValue({
        ...mockDocument,
        status: 'DELETED',
      })
      vi.mocked(CacheService.del).mockRejectedValue(
        new Error('Redis connection error')
      )

      const consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation()

      await handler(req, res)

      expect(res._getStatusCode()).toBe(200)
      expect(JSON.parse(res._getData())).toEqual({
        message: 'Document deleted successfully',
        documentId: 'doc-123',
      })

      expect(consoleWarnSpy).toHaveBeenCalledWith(
        'Failed to clear cache:',
        expect.any(Error)
      )

      consoleWarnSpy.mockRestore()
    })
  })

  describe('错误处理测试', () => {
    it('应该处理文档查询失败', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'DELETE',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockRejectedValue(
        new Error('Database connection error')
      )

      await handler(req, res)

      expect(res._getStatusCode()).toBe(500)
      expect(JSON.parse(res._getData())).toEqual({
        error: 'Failed to delete document',
      })
    })

    it('应该处理文档更新失败', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'DELETE',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockResolvedValue(mockDocument)
      vi.mocked(prisma.document.update).mockRejectedValue(
        new Error('Update failed')
      )

      await handler(req, res)

      expect(res._getStatusCode()).toBe(500)
      expect(JSON.parse(res._getData())).toEqual({
        error: 'Failed to delete document',
      })
    })

    it('应该记录错误日志', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'DELETE',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      const error = new Error('Test error')
      vi.mocked(prisma.document.findFirst).mockRejectedValue(error)

      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation()

      await handler(req, res)

      expect(consoleErrorSpy).toHaveBeenCalledWith('Delete document error:', error)

      consoleErrorSpy.mockRestore()
    })
  })

  describe('并发删除测试', () => {
    it('应该正确处理并发删除请求', async () => {
      const documents = Array.from({ length: 5 }, (_, i) => ({
        id: `doc-${i}`,
        ...mockDocument,
      }))

      const promises = documents.map(async (doc) => {
        const { req, res } = createMocks<AuthenticatedRequest>({
          method: 'DELETE',
          query: { id: doc.id },
          user: mockUser,
        })

        vi.mocked(prisma.document.findFirst).mockResolvedValueOnce(doc)
        vi.mocked(prisma.document.update).mockResolvedValueOnce({
          ...doc,
          status: 'DELETED',
        })
        vi.mocked(CacheService.del).mockResolvedValue(1)

        await handler(req, res)

        return {
          status: res._getStatusCode(),
          data: JSON.parse(res._getData()),
        }
      })

      const results = await Promise.all(promises)

      results.forEach((result, index) => {
        expect(result.status).toBe(200)
        expect(result.data).toEqual({
          message: 'Document deleted successfully',
          documentId: `doc-${index}`,
        })
      })
    })
  })

  describe('特殊场景测试', () => {
    it('应该处理已删除的文档再次删除', async () => {
      const deletedDocument = {
        ...mockDocument,
        status: 'DELETED',
      }

      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'DELETE',
        query: { id: 'doc-123' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockResolvedValue(deletedDocument)
      vi.mocked(prisma.document.update).mockResolvedValue({
        ...deletedDocument,
        updatedAt: new Date(),
      })

      await handler(req, res)

      expect(res._getStatusCode()).toBe(200)
      expect(JSON.parse(res._getData())).toEqual({
        message: 'Document deleted successfully',
        documentId: 'doc-123',
      })
    })

    it('应该处理非标准格式的文档ID', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'DELETE',
        query: { id: '../../etc/passwd' },
        user: mockUser,
      })

      vi.mocked(prisma.document.findFirst).mockResolvedValue(null)

      await handler(req, res)

      expect(res._getStatusCode()).toBe(404)
      expect(JSON.parse(res._getData())).toEqual({
        error: 'Document not found',
      })

      expect(prisma.document.findFirst).toHaveBeenCalledWith({
        where: {
          id: '../../etc/passwd',
          uploadedBy: 'user-123',
        },
      })
    })
  })
})