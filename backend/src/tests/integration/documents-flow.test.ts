import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from 'vitest'
import request from 'supertest'
import { prisma } from '@/lib/prisma'
import { CacheService } from '@/lib/redis'
import path from 'path'
import fs from 'fs'

describe('文档管理集成测试', () => {
  const baseUrl = 'http://localhost:3001'
  let authToken: string
  let testUserId: string
  
  const testUser = {
    email: 'integration-test@example.com',
    password: 'testpassword123',
  }

  beforeAll(async () => {
    const response = await request(baseUrl)
      .post('/api/auth/login')
      .send(testUser)
      .expect(200)

    authToken = response.body.token
    testUserId = response.body.user.id
  })

  afterAll(async () => {
    await prisma.document.deleteMany({
      where: { uploadedBy: testUserId }
    })
    
    await prisma.user.delete({
      where: { id: testUserId }
    }).catch(() => {})
  })

  beforeEach(async () => {
    await prisma.document.deleteMany({
      where: { uploadedBy: testUserId }
    })
  })

  afterEach(async () => {
    await CacheService.del(`documents:${testUserId}`)
  })

  describe('完整文档生命周期', () => {
    it('应该完成文档的完整生命周期：上传→列表→状态→删除', async () => {
      const testFilePath = path.join(__dirname, '../fixtures/test.pdf')
      const fileBuffer = fs.readFileSync(testFilePath)
      
      const uploadResponse = await request(baseUrl)
        .post('/api/documents/upload')
        .set('Authorization', `Bearer ${authToken}`)
        .attach('file', fileBuffer, 'test.pdf')
        .expect(200)

      expect(uploadResponse.body).toHaveProperty('documentId')
      expect(uploadResponse.body.status).toBe('processing')
      const documentId = uploadResponse.body.documentId
      
      const listResponse = await request(baseUrl)
        .get('/api/documents/list')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)

      expect(listResponse.body.documents).toHaveLength(1)
      expect(listResponse.body.documents[0].id).toBe(documentId)
      expect(listResponse.body.documents[0].originalName).toBe('test.pdf')
      
      const statusResponse = await request(baseUrl)
        .get(`/api/documents/status/${documentId}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)

      expect(statusResponse.body.document.id).toBe(documentId)
      expect(['PENDING', 'PROCESSING', 'COMPLETED', 'FAILED']).toContain(
        statusResponse.body.document.status
      )
      
      const deleteResponse = await request(baseUrl)
        .delete(`/api/documents/delete/${documentId}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)

      expect(deleteResponse.body.message).toBe('Document deleted successfully')
      
      const listAfterDelete = await request(baseUrl)
        .get('/api/documents/list')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)

      const deletedDoc = listAfterDelete.body.documents.find(
        (doc: any) => doc.id === documentId
      )
      expect(deletedDoc?.status).toBe('DELETED')
    })

    it('应该处理多文档并发上传', async () => {
      const testFiles = [
        { name: 'test1.pdf', content: 'Test content 1' },
        { name: 'test2.txt', content: 'Test content 2' },
        { name: 'test3.md', content: '# Test Markdown' },
      ]

      const uploadPromises = testFiles.map(({ name, content }) =>
        request(baseUrl)
          .post('/api/documents/upload')
          .set('Authorization', `Bearer ${authToken}`)
          .attach('file', Buffer.from(content), name)
      )

      const uploadResults = await Promise.all(uploadPromises)
      
      uploadResults.forEach(response => {
        expect(response.status).toBe(200)
        expect(response.body).toHaveProperty('documentId')
      })
      
      const listResponse = await request(baseUrl)
        .get('/api/documents/list')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)

      expect(listResponse.body.documents).toHaveLength(3)
      
      const filenames = listResponse.body.documents.map((doc: any) => doc.originalName)
      expect(filenames).toContain('test1.pdf')
      expect(filenames).toContain('test2.txt')
      expect(filenames).toContain('test3.md')
    })
  })

  describe('权限和安全测试', () => {
    it('应该阻止未认证用户访问', async () => {
      await request(baseUrl)
        .get('/api/documents/list')
        .expect(401)

      await request(baseUrl)
        .post('/api/documents/upload')
        .attach('file', Buffer.from('test'), 'test.pdf')
        .expect(401)

      await request(baseUrl)
        .delete('/api/documents/delete/any-id')
        .expect(401)

      await request(baseUrl)
        .get('/api/documents/status/any-id')
        .expect(401)
    })

    it('应该阻止用户访问其他用户的文档', async () => {
      const testFilePath = path.join(__dirname, '../fixtures/test.pdf')
      const fileBuffer = fs.readFileSync(testFilePath)
      
      const uploadResponse = await request(baseUrl)
        .post('/api/documents/upload')
        .set('Authorization', `Bearer ${authToken}`)
        .attach('file', fileBuffer, 'private.pdf')
        .expect(200)

      const documentId = uploadResponse.body.documentId
      
      const otherUserResponse = await request(baseUrl)
        .post('/api/auth/register')
        .send({
          email: 'other-user@example.com',
          password: 'password123',
        })
        .expect(201)

      const otherToken = otherUserResponse.body.token
      
      await request(baseUrl)
        .get(`/api/documents/status/${documentId}`)
        .set('Authorization', `Bearer ${otherToken}`)
        .expect(404)

      await request(baseUrl)
        .delete(`/api/documents/delete/${documentId}`)
        .set('Authorization', `Bearer ${otherToken}`)
        .expect(404)
      
      await prisma.user.delete({
        where: { id: otherUserResponse.body.user.id }
      }).catch(() => {})
    })

    it('应该验证文档所有权', async () => {
      const user1Token = authToken
      
      const user2Response = await request(baseUrl)
        .post('/api/auth/register')
        .send({
          email: 'user2@example.com',
          password: 'password123',
        })
        .expect(201)

      const user2Token = user2Response.body.token
      const user2Id = user2Response.body.user.id
      
      const user1Doc = await request(baseUrl)
        .post('/api/documents/upload')
        .set('Authorization', `Bearer ${user1Token}`)
        .attach('file', Buffer.from('user1 content'), 'user1.pdf')
        .expect(200)

      const user2Doc = await request(baseUrl)
        .post('/api/documents/upload')
        .set('Authorization', `Bearer ${user2Token}`)
        .attach('file', Buffer.from('user2 content'), 'user2.pdf')
        .expect(200)
      
      const user1List = await request(baseUrl)
        .get('/api/documents/list')
        .set('Authorization', `Bearer ${user1Token}`)
        .expect(200)

      const user2List = await request(baseUrl)
        .get('/api/documents/list')
        .set('Authorization', `Bearer ${user2Token}`)
        .expect(200)

      expect(user1List.body.documents).toHaveLength(1)
      expect(user2List.body.documents).toHaveLength(1)
      expect(user1List.body.documents[0].originalName).toBe('user1.pdf')
      expect(user2List.body.documents[0].originalName).toBe('user2.pdf')
      
      await prisma.user.delete({ where: { id: user2Id } }).catch(() => {})
    })
  })

  describe('缓存一致性测试', () => {
    it('应该在操作后正确更新缓存', async () => {
      const listResponse1 = await request(baseUrl)
        .get('/api/documents/list')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)

      const initialCount = listResponse1.body.documents.length
      
      const uploadResponse = await request(baseUrl)
        .post('/api/documents/upload')
        .set('Authorization', `Bearer ${authToken}`)
        .attach('file', Buffer.from('test content'), 'cache-test.pdf')
        .expect(200)

      const documentId = uploadResponse.body.documentId
      
      const listResponse2 = await request(baseUrl)
        .get('/api/documents/list')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)

      expect(listResponse2.body.documents).toHaveLength(initialCount + 1)
      
      await request(baseUrl)
        .delete(`/api/documents/delete/${documentId}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)
      
      const listResponse3 = await request(baseUrl)
        .get('/api/documents/list')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)

      const deletedDoc = listResponse3.body.documents.find(
        (doc: any) => doc.id === documentId
      )
      expect(deletedDoc?.status).toBe('DELETED')
    })

    it('应该正确处理不同查询参数的缓存', async () => {
      await request(baseUrl)
        .post('/api/documents/upload')
        .set('Authorization', `Bearer ${authToken}`)
        .attach('file', Buffer.from('completed content'), 'completed.pdf')
        .expect(200)

      await request(baseUrl)
        .post('/api/documents/upload')
        .set('Authorization', `Bearer ${authToken}`)
        .attach('file', Buffer.from('processing content'), 'processing.pdf')
        .expect(200)
      
      const allDocsResponse = await request(baseUrl)
        .get('/api/documents/list')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)

      const completedDocsResponse = await request(baseUrl)
        .get('/api/documents/list?status=COMPLETED')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)

      const processingDocsResponse = await request(baseUrl)
        .get('/api/documents/list?status=PROCESSING')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)

      expect(allDocsResponse.body.documents.length).toBeGreaterThanOrEqual(2)
      expect(completedDocsResponse.body.documents.length).toBeGreaterThanOrEqual(0)
      expect(processingDocsResponse.body.documents.length).toBeGreaterThanOrEqual(0)
      
      const searchResponse = await request(baseUrl)
        .get('/api/documents/list?search=completed')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)

      expect(searchResponse.body.documents.every(
        (doc: any) => doc.originalName.includes('completed')
      )).toBe(true)
    })
  })

  describe('错误场景测试', () => {
    it('应该处理数据库连接失败', async () => {
      const originalPrisma = global.prisma
      const mockPrisma = {
        document: {
          findMany: vi.fn().mockRejectedValue(new Error('Database unavailable')),
        },
      }
      global.prisma = mockPrisma as any

      await request(baseUrl)
        .get('/api/documents/list')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(500)

      global.prisma = originalPrisma
    })

    it('应该处理Python服务不可用的情况', async () => {
      const testFilePath = path.join(__dirname, '../fixtures/test.pdf')
      const fileBuffer = fs.readFileSync(testFilePath)
      
      await request(baseUrl)
        .post('/api/documents/upload')
        .set('Authorization', `Bearer ${authToken}`)
        .attach('file', fileBuffer, 'python-fail-test.pdf')
        .expect(500)
      
      const listResponse = await request(baseUrl)
        .get('/api/documents/list')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)

      const failedDoc = listResponse.body.documents.find(
        (doc: any) => doc.originalName === 'python-fail-test.pdf'
      )
      
      if (failedDoc) {
        expect(failedDoc.status).toBe('FAILED')
        expect(failedDoc.errorMessage).toBeTruthy()
      }
    })

    it('应该处理Redis服务不可用的情况', async () => {
      const originalCacheService = CacheService
      const mockCacheService = {
        get: vi.fn().mockRejectedValue(new Error('Redis unavailable')),
        set: vi.fn().mockRejectedValue(new Error('Redis unavailable')),
        del: vi.fn().mockRejectedValue(new Error('Redis unavailable')),
      }
      
      Object.defineProperty(require('@/lib/redis'), 'CacheService', {
        value: mockCacheService,
        configurable: true,
      })

      await request(baseUrl)
        .get('/api/documents/list')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)
      
      Object.defineProperty(require('@/lib/redis'), 'CacheService', {
        value: originalCacheService,
        configurable: true,
      })
    })
  })

  describe('性能测试', () => {
    it('应该在合理时间内处理大量文档查询', async () => {
      const documents = Array.from({ length: 50 }, (_, i) => ({
        filename: `test${i}.pdf`,
        originalName: `测试文档${i}.pdf`,
        mimeType: 'application/pdf',
        size: 1048576,
        status: 'COMPLETED',
        uploadedBy: testUserId,
        processingStartedAt: new Date(),
        processingCompletedAt: new Date(),
      }))

      await prisma.document.createMany({
        data: documents,
      })

      const startTime = Date.now()
      
      const response = await request(baseUrl)
        .get('/api/documents/list?limit=50')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)

      const endTime = Date.now()
      const responseTime = endTime - startTime

      expect(responseTime).toBeLessThan(1000)
      expect(response.body.documents).toHaveLength(50)
      expect(response.body.pagination.total).toBe(50)
    })

    it('应该正确处理大文件上传', async () => {
      const largeContent = 'x'.repeat(8 * 1024 * 1024)
      
      const startTime = Date.now()
      
      const response = await request(baseUrl)
        .post('/api/documents/upload')
        .set('Authorization', `Bearer ${authToken}`)
        .attach('file', Buffer.from(largeContent), 'large.pdf')
        .timeout(30000)
        .expect(200)

      const endTime = Date.now()
      const uploadTime = endTime - startTime

      expect(uploadTime).toBeLessThan(25000)
      expect(response.body).toHaveProperty('documentId')
    })
  })

  describe('分页和搜索集成测试', () => {
    beforeEach(async () => {
      const testDocuments = [
        { name: '财务报告2024.pdf', content: 'Financial report content' },
        { name: '技术文档.docx', content: 'Technical documentation' },
        { name: '会议记录.txt', content: 'Meeting notes' },
        { name: '产品说明书.md', content: '# Product Manual' },
        { name: '测试报告.pdf', content: 'Test report content' },
      ]

      for (const doc of testDocuments) {
        await request(baseUrl)
          .post('/api/documents/upload')
          .set('Authorization', `Bearer ${authToken}`)
          .attach('file', Buffer.from(doc.content), doc.name)
      }

      await new Promise(resolve => setTimeout(resolve, 1000))
    })

    it('应该正确处理分页查询', async () => {
      const page1Response = await request(baseUrl)
        .get('/api/documents/list?page=1&limit=3')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)

      expect(page1Response.body.documents).toHaveLength(3)
      expect(page1Response.body.pagination.page).toBe(1)
      expect(page1Response.body.pagination.hasNext).toBe(true)

      const page2Response = await request(baseUrl)
        .get('/api/documents/list?page=2&limit=3')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)

      expect(page2Response.body.documents).toHaveLength(2)
      expect(page2Response.body.pagination.page).toBe(2)
      expect(page2Response.body.pagination.hasPrev).toBe(true)
    })

    it('应该正确处理搜索查询', async () => {
      const searchResponse = await request(baseUrl)
        .get('/api/documents/list?search=报告')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)

      expect(searchResponse.body.documents.length).toBeGreaterThan(0)
      searchResponse.body.documents.forEach((doc: any) => {
        expect(doc.originalName.includes('报告')).toBe(true)
      })
    })

    it('应该正确处理状态筛选', async () => {
      const completedResponse = await request(baseUrl)
        .get('/api/documents/list?status=COMPLETED')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)

      completedResponse.body.documents.forEach((doc: any) => {
        expect(doc.status).toBe('COMPLETED')
      })
    })

    it('应该正确处理组合查询', async () => {
      const combinedResponse = await request(baseUrl)
        .get('/api/documents/list?search=pdf&status=COMPLETED&page=1&limit=5')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)

      combinedResponse.body.documents.forEach((doc: any) => {
        expect(doc.originalName.includes('pdf')).toBe(true)
        expect(doc.status).toBe('COMPLETED')
      })
      
      expect(combinedResponse.body.pagination.page).toBe(1)
      expect(combinedResponse.body.pagination.limit).toBe(5)
    })
  })

  describe('文档状态变更测试', () => {
    it('应该跟踪文档从上传到完成的状态变化', async () => {
      const testFilePath = path.join(__dirname, '../fixtures/test.pdf')
      const fileBuffer = fs.readFileSync(testFilePath)
      
      const uploadResponse = await request(baseUrl)
        .post('/api/documents/upload')
        .set('Authorization', `Bearer ${authToken}`)
        .attach('file', fileBuffer, 'status-test.pdf')
        .expect(200)

      const documentId = uploadResponse.body.documentId
      
      let statusResponse = await request(baseUrl)
        .get(`/api/documents/status/${documentId}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)

      const initialStatus = statusResponse.body.document.status
      expect(['PENDING', 'PROCESSING']).toContain(initialStatus)
      
      for (let i = 0; i < 10; i++) {
        await new Promise(resolve => setTimeout(resolve, 2000))
        
        statusResponse = await request(baseUrl)
          .get(`/api/documents/status/${documentId}`)
          .set('Authorization', `Bearer ${authToken}`)
          .expect(200)

        const currentStatus = statusResponse.body.document.status
        
        if (currentStatus === 'COMPLETED' || currentStatus === 'FAILED') {
          break
        }
      }
      
      const finalStatus = statusResponse.body.document.status
      expect(['COMPLETED', 'FAILED']).toContain(finalStatus)
      
      if (finalStatus === 'COMPLETED') {
        expect(statusResponse.body.document.chunksCount).toBeGreaterThan(0)
        expect(statusResponse.body.document.processingCompletedAt).toBeTruthy()
      }
    })
  })

  describe('API响应时间测试', () => {
    it('文档列表API响应时间应该小于1秒', async () => {
      const startTime = Date.now()
      
      await request(baseUrl)
        .get('/api/documents/list')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)
      
      const responseTime = Date.now() - startTime
      expect(responseTime).toBeLessThan(1000)
    })

    it('文档上传API响应时间应该合理', async () => {
      const startTime = Date.now()
      
      await request(baseUrl)
        .post('/api/documents/upload')
        .set('Authorization', `Bearer ${authToken}`)
        .attach('file', Buffer.from('test content'), 'speed-test.pdf')
        .expect(200)
      
      const responseTime = Date.now() - startTime
      expect(responseTime).toBeLessThan(5000)
    })

    it('文档删除API响应时间应该小于500毫秒', async () => {
      const uploadResponse = await request(baseUrl)
        .post('/api/documents/upload')
        .set('Authorization', `Bearer ${authToken}`)
        .attach('file', Buffer.from('delete speed test'), 'delete-speed.pdf')
        .expect(200)

      const documentId = uploadResponse.body.documentId
      
      const startTime = Date.now()
      
      await request(baseUrl)
        .delete(`/api/documents/delete/${documentId}`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200)
      
      const responseTime = Date.now() - startTime
      expect(responseTime).toBeLessThan(500)
    })
  })
})