export interface Document {
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
  vectorId: string
  metadata?: Record<string, any>
}