import type { QueryRequest, QueryResponse, PaginatedResponse, QueryHistory } from '@/types'
import { apiClient } from './api'

export const queryService = {
  async submitQuery(data: QueryRequest): Promise<QueryResponse> {
    return apiClient.post('/api/query', data)
  },

  async getQueryHistory(params: {
    page?: number
    limit?: number
    status?: string
  }): Promise<PaginatedResponse<QueryHistory>> {
    return apiClient.get('/api/query/history', { params })
  },

  async chatQuery(messages: Array<{ role: string; content: string }>): Promise<any> {
    return apiClient.post('/api/query/chat', { messages })
  },

  async clearAllHistory(): Promise<void> {
    return apiClient.delete('/api/query/history-clear', {
      data: { confirm: true }
    })
  },

  async getHistoryStats(): Promise<{ total: number }> {
    const response = await apiClient.get('/api/query/history-stats')
    return response
  },
}