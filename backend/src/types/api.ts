// 统一响应格式
export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  error?: string
  message?: string
  timestamp: string
  requestId?: string
}

// 分页响应格式
export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  pagination: {
    page: number
    limit: number
    total: number
    totalPages: number
    hasNext: boolean
    hasPrev: boolean
  }
}

// 错误响应格式
export interface ErrorResponse {
  success: false
  error: string
  details?: any
  code?: string
  timestamp: string
}

// 文档相关类型
export interface DocumentUploadRequest {
  file: File
  metadata?: Record<string, any>
}

export interface DocumentStatus {
  id: string
  filename: string
  status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED' | 'DELETED'
  progress?: number
  error?: string
}

// 查询相关类型
export interface QueryRequest {
  query: string
  topK?: number
  useCache?: boolean
}

export interface QueryResponse {
  answer: string
  sources: {
    id: string
    content: string
    score: number
    metadata?: Record<string, any>
  }[]
  responseTime: number
  cached?: boolean
}

// 用户相关类型
export interface UserProfile {
  id: string
  email: string
  name?: string
  role: 'USER' | 'ADMIN' | 'SUPER_ADMIN'
  avatar?: string
  isActive: boolean
  createdAt: string
  updatedAt: string
}