import type { 
  DocumentListParams,
  DocumentListResponse,
  DocumentStatusResponse,
} from '@/types'
import { apiClient } from './api'

export class DocumentService {
  private static instance: DocumentService
  
  static getInstance(): DocumentService {
    if (!DocumentService.instance) {
      DocumentService.instance = new DocumentService()
    }
    return DocumentService.instance
  }

  async uploadDocument(
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<{ documentId: string; status: string }> {
    const formData = new FormData()
    formData.append('file', file)
    
    const response = await apiClient.post('/api/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          )
          onProgress(progress)
        }
      }
    })
    return response.data
  }

  async getDocuments(params: DocumentListParams): Promise<DocumentListResponse> {
    const response = await apiClient.get('/api/documents/list', { params })
    return response
  }

  async getDocumentStatus(id: string): Promise<DocumentStatusResponse> {
    const response = await apiClient.get(`/api/documents/status/${id}`)
    return response
  }

  async deleteDocument(id: string): Promise<{ message: string; documentId: string }> {
    const response = await apiClient.delete(`/api/documents/delete/${id}`)
    return response
  }

  async batchDeleteDocuments(documentIds: string[]): Promise<{
    succeeded: number
    failed: number
  }> {
    // 并发删除，最大并发数5
    const chunks = this.chunkArray(documentIds, 5)
    const results = []
    
    for (const chunk of chunks) {
      const promises = chunk.map(id => this.deleteDocument(id).catch(err => ({ error: err })))
      const chunkResults = await Promise.allSettled(promises)
      results.push(...chunkResults)
    }
    
    const succeeded = results.filter(r => r.status === 'fulfilled').length
    const failed = results.filter(r => r.status === 'rejected').length
    
    return { succeeded, failed }
  }

  private chunkArray<T>(array: T[], size: number): T[][] {
    const chunks: T[][] = []
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size))
    }
    return chunks
  }
}

export const documentService = DocumentService.getInstance()