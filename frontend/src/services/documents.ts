import type { Document, DocumentUploadRequest, PaginatedResponse } from '@/types'
import { apiClient } from './api'

export const documentService = {
  async uploadDocument(data: DocumentUploadRequest): Promise<{ documentId: string }> {
    const formData = new FormData()
    formData.append('file', data.file)
    
    return apiClient.post('/api/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
  },

  async getDocumentList(params: {
    page?: number
    limit?: number
    status?: string
    search?: string
  }): Promise<PaginatedResponse<Document>> {
    return apiClient.get('/api/documents/list', { params })
  },

  async getDocumentStatus(id: string): Promise<{ document: Document }> {
    return apiClient.get(`/api/documents/status/${id}`)
  },

  async deleteDocument(id: string): Promise<void> {
    return apiClient.delete(`/api/documents/delete/${id}`)
  },
}