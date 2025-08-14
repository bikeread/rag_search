import type { QueryRequest, QueryResponse, PaginatedResponse, QueryHistory } from '@/types'
import { apiClient } from './api'

export const queryService = {
  async submitQuery(data: QueryRequest): Promise<QueryResponse> {
    // 调用后端API的/api/query端点
    return apiClient.post('/api/query', data)
  },

  async getQueryHistory(params: {
    page?: number
    limit?: number
    status?: string
  }): Promise<PaginatedResponse<QueryHistory>> {
    // 模拟空的查询历史，因为RAG服务不提供历史功能
    return Promise.resolve({
      data: [],
      pagination: {
        total: 0,
        page: 1,
        limit: 10,
        totalPages: 0
      }
    })
  },

  async chatQuery(messages: Array<{ role: string; content: string }>): Promise<any> {
    // 转换聊天格式为RAG查询格式
    const lastMessage = messages[messages.length - 1]
    if (lastMessage && lastMessage.role === 'user') {
      return this.submitQuery({ 
        query: lastMessage.content,
        top_k: 3
      })
    }
    throw new Error('无效的聊天消息格式')
  },

  async clearAllHistory(): Promise<void> {
    // 模拟成功，因为RAG服务不提供历史清空功能
    return Promise.resolve()
  },

  async getHistoryStats(): Promise<{ total: number }> {
    // 模拟空的统计数据
    return Promise.resolve({ total: 0 })
  },
}