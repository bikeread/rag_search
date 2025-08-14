export interface DocumentModel {
  id: string
  filename: string
  originalName: string
  mimeType: string
  size: number
  status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED' | 'DELETED'
  chunksCount: number
  createdAt: string
  updatedAt: string
  processingStartedAt?: string
  processingCompletedAt?: string
  errorMessage?: string
}

export interface DocumentUploadRequest {
  file: File
}

export interface DocumentChunk {
  id: string
  content: string
  chunkIndex: number
  vectorId?: string
  metadata?: Record<string, any>
  createdAt: string
}

export interface DocumentListParams {
  page?: number
  limit?: number
  status?: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED'
  search?: string
}

export interface DocumentStatusResponse {
  document: DocumentModel
  processingStatus?: {
    currentStep: string
    progress: number
    estimatedTime?: number
  }
  chunks: DocumentChunk[]
}

export interface DocumentListResponse {
  documents: DocumentModel[]
  pagination: {
    page: number
    limit: number
    total: number
    totalPages: number
    hasNext: boolean
    hasPrev: boolean
  }
}