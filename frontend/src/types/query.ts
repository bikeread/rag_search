export interface QueryRequest {
  query: string
  topK?: number
  useCache?: boolean
}

export interface QueryResponse {
  queryId: string
  answer: string
  sources: QuerySource[]
  responseTime: number
  metadata: {
    topK: number
    searchTime: number
    sourcesCount: number
  }
  cached?: boolean
}

export interface QuerySource {
  id: string
  content: string
  score: number
  metadata?: Record<string, any>
}

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  sources?: QuerySource[]
}

export interface QueryHistory {
  id: string
  text: string
  response?: string
  status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED'
  responseTime?: number
  createdAt: string
  metadata?: Record<string, any>
}