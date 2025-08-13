import { createMocks } from 'node-mocks-http'
import { NextApiRequest, NextApiResponse } from 'next'
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { prismaMock } from '../../__mocks__/prisma'
import { ragService } from '@/services/pythonServices'
import handler from '@/pages/api/query/chat'
import { AuthenticatedRequest } from '@/lib/jwtAuth'

// Mock dependencies
vi.mock('@/lib/prisma', () => ({
  prisma: prismaMock,
}))

vi.mock('@/services/pythonServices', () => ({
  ragService: {
    chat: vi.fn(),
  },
}))

vi.mock('@/lib/cors', () => ({
  withCorsAndAuth: (handler: any) => handler,
}))

const mockRagService = ragService as any

describe('/api/query/chat', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    prismaMock.chatSession.create.mockResolvedValue({
      id: 'test-session-id',
      userId: 'test-user-id',
      title: 'Test Chat',
      createdAt: new Date(),
      updatedAt: new Date(),
    })
    prismaMock.chatMessage.create.mockResolvedValue({
      id: 'test-message-id',
      sessionId: 'test-session-id',
      role: 'assistant',
      content: 'Test response',
      createdAt: new Date(),
      updatedAt: new Date(),
    })
  })

  describe('POST /api/query/chat', () => {
    it('应该成功处理聊天请求并返回结果', async () => {
      mockRagService.chat.mockResolvedValue({
        response: 'AI助手的回复内容',
        context: 'some context',
        sources: [],
      })

      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'POST',
        body: {
          messages: [
            {
              role: 'user',
              content: '你好，什么是RAG？'
            }
          ],
        },
      })

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
        response: 'AI助手的回复内容',
        context: 'some context',
        sources: [],
      })

      // Verify RAG service was called
      expect(mockRagService.chat).toHaveBeenCalledWith([
        {
          role: 'user',
          content: '你好，什么是RAG？'
        }
      ])
    })

    it('应该验证消息格式并拒绝无效请求', async () => {
      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'POST',
        body: {
          messages: [
            {
              role: 'invalid-role', // 无效角色
              content: 'test content'
            }
          ],
        },
      })

      req.user = {
        id: 'test-user-id',
        email: 'test@example.com',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(400)
      const data = JSON.parse(res._getData())
      expect(data.error).toBe('Invalid chat data')
      expect(data.details).toBeDefined()
    })

    it('应该处理多轮对话', async () => {
      mockRagService.chat.mockResolvedValue({
        response: '这是多轮对话的回复',
        conversation_id: 'conv-123',
        message_id: 'msg-789',
      })

      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'POST',
        body: {
          messages: [
            {
              role: 'user',
              content: '什么是RAG？'
            },
            {
              role: 'assistant',
              content: 'RAG是检索增强生成技术'
            },
            {
              role: 'user',
              content: '能详细解释一下吗？'
            }
          ],
        },
      })

      req.user = {
        id: 'test-user-id',
        email: 'test@example.com',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(200)
      const data = JSON.parse(res._getData())
      expect(data.response).toBe('这是多轮对话的回复')

      // Verify all messages were sent to RAG service
      expect(mockRagService.chat).toHaveBeenCalledWith([
        { role: 'user', content: '什么是RAG？' },
        { role: 'assistant', content: 'RAG是检索增强生成技术' },
        { role: 'user', content: '能详细解释一下吗？' }
      ])
    })

    it('应该处理RAG聊天服务错误', async () => {
      mockRagService.chat.mockRejectedValue(new Error('Chat service error'))

      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'POST',
        body: {
          messages: [
            {
              role: 'user',
              content: '测试消息'
            }
          ],
        },
      })

      req.user = {
        id: 'test-user-id',
        email: 'test@example.com',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(500)
      const data = JSON.parse(res._getData())
      expect(data.error).toBe('Chat failed')
    })

    it('应该拒绝空消息列表', async () => {
      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'POST',
        body: {
          messages: [], // 空消息列表
        },
      })

      req.user = {
        id: 'test-user-id',
        email: 'test@example.com',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(400)
      const data = JSON.parse(res._getData())
      expect(data.error).toBe('Invalid chat data')
    })

    it('应该拒绝过多消息的请求', async () => {
      // 创建超过50条消息的数组
      const tooManyMessages = Array.from({ length: 51 }, (_, i) => ({
        role: i % 2 === 0 ? 'user' : 'assistant',
        content: `Message ${i}`
      }))

      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'POST',
        body: {
          messages: tooManyMessages,
        },
      })

      req.user = {
        id: 'test-user-id',
        email: 'test@example.com',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(400)
      expect(mockRagService.chat).not.toHaveBeenCalled()
    })

    it('应该拒绝包含空内容消息的请求', async () => {
      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'POST',
        body: {
          messages: [
            {
              role: 'user',
              content: '' // 空内容
            }
          ],
        },
      })

      req.user = {
        id: 'test-user-id',
        email: 'test@example.com',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(400)
      expect(mockRagService.chat).not.toHaveBeenCalled()
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

    it('应该处理网络超时错误', async () => {
      const timeoutError = new Error('Request timeout')
      timeoutError.name = 'TIMEOUT'
      mockRagService.chat.mockRejectedValue(timeoutError)

      const { req, res } = createMocks<AuthenticatedRequest, NextApiResponse>({
        method: 'POST',
        body: {
          messages: [
            {
              role: 'user',
              content: '这是一个可能导致超时的长查询...'
            }
          ],
        },
      })

      req.user = {
        id: 'test-user-id',
        email: 'test@example.com',
      }

      await handler(req, res)

      expect(res._getStatusCode()).toBe(500)
      const data = JSON.parse(res._getData())
      expect(data.error).toBe('Chat failed')
      expect(data.details).toContain('Request timeout')
    })
  })
})