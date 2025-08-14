import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { createMocks } from 'node-mocks-http'
import formidable from 'formidable'
import fs from 'fs/promises'
import handler from '@/pages/api/documents/upload'
import { prisma } from '@/lib/prisma'
import { documentProcessor } from '@/services/pythonServices'
import { CacheService } from '@/lib/redis'
import { withTransaction } from '@/lib/transaction'
import type { AuthenticatedRequest } from '@/lib/jwtAuth'

// Mocks are already in setup.ts

describe('/api/documents/upload', () => {
  const mockUser = {
    id: 'user-123',
    email: 'test@example.com',
  }

  const mockFile = {
    originalFilename: '测试文档.pdf',
    mimetype: 'application/pdf',
    size: 1048576,
    filepath: '/tmp/upload_123456',
  }

  const mockDocument = {
    id: 'doc-123',
    filename: '测试文档.pdf',
    originalName: '测试文档.pdf',
    mimeType: 'application/pdf',
    size: 1048576,
    status: 'PENDING',
    uploadedBy: 'user-123',
    processingStartedAt: new Date(),
    createdAt: new Date(),
    updatedAt: new Date(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.resetAllMocks()
  })

  describe('文件上传测试', () => {
    it('应该成功上传PDF文件', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'POST',
        user: mockUser,
      })

      const mockForm = {
        parse: vi.fn().mockResolvedValue([{}, { file: mockFile }]),
      }
      vi.mocked(formidable).mockReturnValue(mockForm as any)
      
      // Mock withTransaction to actually call the callback with proper transaction object
      vi.mocked(withTransaction).mockImplementation((callback) => 
        callback({
          document: {
            create: vi.mocked(prisma.document.create),
            update: vi.mocked(prisma.document.update),
          }
        })
      )
      
      vi.mocked(prisma.document.create).mockResolvedValue(mockDocument)
      vi.mocked(fs.readFile).mockResolvedValue(Buffer.from('file content'))
      vi.mocked(documentProcessor.uploadDocument).mockResolvedValue({
        status: 'processing',
        taskId: 'task-123',
      })
      vi.mocked(prisma.document.update).mockResolvedValue({
        ...mockDocument,
        status: 'PROCESSING',
      })
      
      // Mock cache service
      vi.mocked(CacheService.keys).mockResolvedValue(['key1', 'key2'])
      vi.mocked(CacheService.del).mockResolvedValue(undefined)

      await handler(req, res)

      expect(res._getStatusCode()).toBe(201)
      const responseData = JSON.parse(res._getData())
      
      expect(responseData.success).toBe(true)
      expect(responseData.message).toBe('Document uploaded successfully')
      expect(responseData.data).toBeDefined()
      expect(responseData.data.documentId).toBe('doc-123')
    })

    it('应该拒绝非POST请求', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'GET',
        user: mockUser,
      })

      await handler(req, res)

      expect(res._getStatusCode()).toBe(405)
      expect(JSON.parse(res._getData())).toEqual({
        error: 'Method not allowed',
      })
    })

    it('应该处理没有文件的情况', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'POST',
        user: mockUser,
      })

      const mockForm = {
        parse: vi.fn().mockResolvedValue([{}, {}]),
      }
      vi.mocked(formidable).mockReturnValue(mockForm as any)

      await handler(req, res)

      expect(res._getStatusCode()).toBe(400)
      const responseData = JSON.parse(res._getData())
      expect(responseData.success).toBe(false)
      expect(responseData.error).toBe('No file provided')
    })
  })

  describe('文件类型验证测试', () => {
    const allowedTypes = [
      { mimetype: 'application/pdf', name: 'PDF' },
      { mimetype: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', name: 'DOCX' },
      { mimetype: 'application/msword', name: 'DOC' },
      { mimetype: 'text/plain', name: 'TXT' },
      { mimetype: 'text/markdown', name: 'MD' },
    ]

    allowedTypes.forEach(({ mimetype, name }) => {
      it(`应该接受${name}文件类型`, async () => {
        const { req, res } = createMocks<AuthenticatedRequest>({
          method: 'POST',
          user: mockUser,
        })

        const testFile = { ...mockFile, mimetype }
        const mockForm = {
          parse: vi.fn().mockResolvedValue([{}, { file: testFile }]),
        }
        vi.mocked(formidable).mockReturnValue(mockForm as any)
        vi.mocked(prisma.document.create).mockResolvedValue(mockDocument)
        vi.mocked(fs.readFile).mockResolvedValue(Buffer.from('file content'))
        vi.mocked(documentProcessor.uploadDocument).mockResolvedValue({
          status: 'processing',
        })
        vi.mocked(prisma.document.update).mockResolvedValue({
          ...mockDocument,
          status: 'PROCESSING',
        })
        
        // Mock cache service
        vi.mocked(CacheService.keys).mockResolvedValue([])
        vi.mocked(CacheService.del).mockResolvedValue(undefined)

        await handler(req, res)

        expect(res._getStatusCode()).toBe(201)
      })
    })

    it('应该拒绝不支持的文件类型', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'POST',
        user: mockUser,
      })

      const invalidFile = { ...mockFile, mimetype: 'image/jpeg' }
      
      // formidable的filter会过滤掉不支持的文件类型，导致file为undefined
      vi.mocked(formidable).mockImplementation((options: any) => {
        // 验证filter函数正确拒绝了jpeg文件
        const result = options.filter({ mimetype: 'image/jpeg' })
        expect(result).toBe(false)
        
        return {
          parse: vi.fn().mockResolvedValue([{}, {}]), // 空的files对象，没有file字段
        } as any
      })

      await handler(req, res)

      expect(res._getStatusCode()).toBe(400)
      const responseData = JSON.parse(res._getData())
      expect(responseData.success).toBe(false)
      expect(responseData.error).toBe('No file provided')
    })
  })

  describe('文件大小限制测试', () => {
    it('应该使用默认的50MB文件大小限制', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'POST',
        user: mockUser,
      })

      vi.mocked(formidable).mockImplementation((options: any) => {
        expect(options.maxFileSize).toBe(50 * 1024 * 1024) // 50MB
        return {
          parse: vi.fn().mockResolvedValue([{}, { file: mockFile }]),
        } as any
      })

      vi.mocked(prisma.document.create).mockResolvedValue(mockDocument)
      vi.mocked(fs.readFile).mockResolvedValue(Buffer.from('file content'))
      vi.mocked(documentProcessor.uploadDocument).mockResolvedValue({
        status: 'processing',
      })
      vi.mocked(prisma.document.update).mockResolvedValue({
        ...mockDocument,
        status: 'PROCESSING',
      })
      
      // Mock cache service
      vi.mocked(CacheService.keys).mockResolvedValue([])
      vi.mocked(CacheService.del).mockResolvedValue(undefined)

      await handler(req, res)

      expect(formidable).toHaveBeenCalledWith(
        expect.objectContaining({
          maxFileSize: 50 * 1024 * 1024,
        })
      )
    })

    it('应该使用环境变量配置的文件大小限制', async () => {
      const originalEnv = process.env.MAX_FILE_SIZE
      process.env.MAX_FILE_SIZE = '5242880'

      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'POST',
        user: mockUser,
      })

      vi.mocked(formidable).mockImplementation((options: any) => {
        expect(options.maxFileSize).toBe(5242880)
        return {
          parse: vi.fn().mockResolvedValue([{}, { file: mockFile }]),
        } as any
      })

      vi.mocked(prisma.document.create).mockResolvedValue(mockDocument)
      vi.mocked(fs.readFile).mockResolvedValue(Buffer.from('file content'))
      vi.mocked(documentProcessor.uploadDocument).mockResolvedValue({
        status: 'processing',
      })
      vi.mocked(prisma.document.update).mockResolvedValue({
        ...mockDocument,
        status: 'PROCESSING',
      })
      
      // Mock cache service
      vi.mocked(CacheService.keys).mockResolvedValue([])
      vi.mocked(CacheService.del).mockResolvedValue(undefined)

      await handler(req, res)

      process.env.MAX_FILE_SIZE = originalEnv
    })
  })

  describe('Python服务集成测试', () => {
    it('应该处理Python服务处理失败', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'POST',
        user: mockUser,
      })

      const mockForm = {
        parse: vi.fn().mockResolvedValue([{}, { file: mockFile }]),
      }
      vi.mocked(formidable).mockReturnValue(mockForm as any)
      vi.mocked(prisma.document.create).mockResolvedValue(mockDocument)
      vi.mocked(fs.readFile).mockResolvedValue(Buffer.from('file content'))
      vi.mocked(documentProcessor.uploadDocument).mockRejectedValue(
        new Error('Processing service unavailable')
      )

      await handler(req, res)

      expect(res._getStatusCode()).toBe(500)
      const responseData = JSON.parse(res._getData())
      expect(responseData.success).toBe(false)
      expect(responseData.error).toBe('Internal server error')
      expect(responseData.code).toBe('INTERNAL_ERROR')
    })

    it('应该在处理成功后更新文档状态', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'POST',
        user: mockUser,
      })

      const mockForm = {
        parse: vi.fn().mockResolvedValue([{}, { file: mockFile }]),
      }
      vi.mocked(formidable).mockReturnValue(mockForm as any)
      vi.mocked(prisma.document.create).mockResolvedValue(mockDocument)
      vi.mocked(fs.readFile).mockResolvedValue(Buffer.from('file content'))
      vi.mocked(documentProcessor.uploadDocument).mockResolvedValue({
        status: 'processing',
        taskId: 'task-123',
      })
      vi.mocked(prisma.document.update).mockResolvedValue({
        ...mockDocument,
        status: 'PROCESSING',
      })
      
      // Mock cache service
      vi.mocked(CacheService.keys).mockResolvedValue([])
      vi.mocked(CacheService.del).mockResolvedValue(undefined)

      await handler(req, res)

      expect(res._getStatusCode()).toBe(201)
      const responseData = JSON.parse(res._getData())
      expect(responseData.success).toBe(true)
      expect(responseData.message).toBe('Document uploaded successfully')
    })
  })

  describe('数据验证测试', () => {
    it('应该验证文件数据的完整性', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'POST',
        user: mockUser,
      })

      const invalidFile = {
        ...mockFile,
        size: null, // 无效的文件大小
      }

      const mockForm = {
        parse: vi.fn().mockResolvedValue([{}, { file: invalidFile }]),
      }
      vi.mocked(formidable).mockReturnValue(mockForm as any)

      await handler(req, res)

      expect(res._getStatusCode()).toBe(400)
      const responseData = JSON.parse(res._getData())
      expect(responseData.success).toBe(false)
      expect(responseData.error).toBe('Invalid file')
      expect(responseData.details).toBeDefined()
    })

    it('应该处理文件名缺失的情况', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'POST',
        user: mockUser,
      })

      const fileWithoutName = {
        ...mockFile,
        originalFilename: null,
      }

      const mockForm = {
        parse: vi.fn().mockResolvedValue([{}, { file: fileWithoutName }]),
      }
      vi.mocked(formidable).mockReturnValue(mockForm as any)

      await handler(req, res)

      // 文件名为null会导致Zod验证失败
      expect(res._getStatusCode()).toBe(400)
      const responseData = JSON.parse(res._getData())
      expect(responseData.success).toBe(false)
      expect(responseData.error).toBe('Invalid file')
    })
  })

  describe('错误处理测试', () => {
    it('应该处理文件读取错误', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'POST',
        user: mockUser,
      })

      const mockForm = {
        parse: vi.fn().mockResolvedValue([{}, { file: mockFile }]),
      }
      vi.mocked(formidable).mockReturnValue(mockForm as any)
      vi.mocked(prisma.document.create).mockResolvedValue(mockDocument)
      vi.mocked(fs.readFile).mockRejectedValue(new Error('File read error'))

      await handler(req, res)

      expect(res._getStatusCode()).toBe(500)
      const responseData = JSON.parse(res._getData())
      expect(responseData.success).toBe(false)
      expect(responseData.error).toBe('Internal server error')
      expect(responseData.code).toBe('INTERNAL_ERROR')
    })

    it('应该处理数据库创建失败', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'POST',
        user: mockUser,
      })

      const mockForm = {
        parse: vi.fn().mockResolvedValue([{}, { file: mockFile }]),
      }
      vi.mocked(formidable).mockReturnValue(mockForm as any)
      vi.mocked(prisma.document.create).mockRejectedValue(
        new Error('Database error')
      )

      await handler(req, res)

      expect(res._getStatusCode()).toBe(500)
      const responseData = JSON.parse(res._getData())
      expect(responseData.success).toBe(false)
      expect(responseData.error).toBe('Internal server error')
      expect(responseData.code).toBe('INTERNAL_ERROR')
    })

    it('应该处理表单解析错误', async () => {
      const { req, res } = createMocks<AuthenticatedRequest>({
        method: 'POST',
        user: mockUser,
      })

      const mockForm = {
        parse: vi.fn().mockRejectedValue(new Error('Form parse error')),
      }
      vi.mocked(formidable).mockReturnValue(mockForm as any)

      await handler(req, res)

      expect(res._getStatusCode()).toBe(500)
      const responseData = JSON.parse(res._getData())
      expect(responseData.success).toBe(false)
      expect(responseData.error).toBe('Internal server error')
      expect(responseData.code).toBe('INTERNAL_ERROR')
    })
  })

  describe('并发上传测试', () => {
    it('应该正确处理多个并发上传请求', async () => {
      const uploads = Array.from({ length: 5 }, (_, i) => ({
        file: { ...mockFile, originalFilename: `test${i}.pdf` },
        documentId: `doc-${i}`,
      }))

      const promises = uploads.map(async ({ file, documentId }) => {
        const { req, res } = createMocks<AuthenticatedRequest>({
          method: 'POST',
          user: mockUser,
        })

        const mockForm = {
          parse: vi.fn().mockResolvedValue([{}, { file }]),
        }
        vi.mocked(formidable).mockReturnValue(mockForm as any)
        vi.mocked(prisma.document.create).mockResolvedValue({
          ...mockDocument,
          id: documentId,
          filename: file.originalFilename,
        })
        vi.mocked(fs.readFile).mockResolvedValue(Buffer.from('file content'))
        vi.mocked(documentProcessor.uploadDocument).mockResolvedValue({
          status: 'processing',
        })
        vi.mocked(prisma.document.update).mockResolvedValue({
          ...mockDocument,
          id: documentId,
          status: 'PROCESSING',
        })
        
        // Mock cache service
        vi.mocked(CacheService.keys).mockResolvedValue([])
        vi.mocked(CacheService.del).mockResolvedValue(undefined)

        await handler(req, res)

        return res._getStatusCode()
      })

      const results = await Promise.all(promises)
      results.forEach((status) => {
        expect(status).toBe(201)
      })
    })
  })
})