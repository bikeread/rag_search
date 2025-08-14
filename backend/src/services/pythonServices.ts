import axios, { AxiosInstance, AxiosRequestConfig } from 'axios'

// 服务配置
const SERVICE_URLS = {
  DOCUMENT_PROCESSOR: process.env.DOCUMENT_PROCESSOR_URL || 'http://localhost:8001',
  VECTOR_SERVICE: process.env.VECTOR_SERVICE_URL || 'http://localhost:8002',
  RAG_SERVICE: process.env.RAG_SERVICE_URL || 'http://localhost:8003',
}

// 通用HTTP客户端类
class PythonServiceClient {
  private client: AxiosInstance

  constructor(baseURL: string, timeout: number = 30000) {
    this.client = axios.create({
      baseURL,
      timeout,
      headers: {
        'Content-Type': 'application/json',
      },
    })

    // 请求拦截器
    this.client.interceptors.request.use(
      (config) => {
        console.log(`[${new Date().toISOString()}] ${config.method?.toUpperCase()} ${config.url}`)
        return config
      },
      (error) => Promise.reject(error)
    )

    // 响应拦截器
    this.client.interceptors.response.use(
      (response) => {
        console.log(`[${new Date().toISOString()}] Response: ${response.status}`)
        return response
      },
      (error) => {
        console.error(`[${new Date().toISOString()}] Error: ${error.message}`)
        return Promise.reject(error)
      }
    )
  }

  async get(url: string, config?: AxiosRequestConfig) {
    const response = await this.client.get(url, config)
    return response.data
  }

  async post(url: string, data?: any, config?: AxiosRequestConfig) {
    const response = await this.client.post(url, data, config)
    return response.data
  }

  async put(url: string, data?: any, config?: AxiosRequestConfig) {
    const response = await this.client.put(url, data, config)
    return response.data
  }

  async delete(url: string, config?: AxiosRequestConfig) {
    const response = await this.client.delete(url, config)
    return response.data
  }
}

// 文档处理服务客户端
export class DocumentProcessorClient extends PythonServiceClient {
  constructor() {
    super(SERVICE_URLS.DOCUMENT_PROCESSOR)
  }

  async uploadDocument(file: Buffer, filename: string, documentId: string, mimeType?: string) {
    const formData = new FormData()
    // 根据文件扩展名推断MIME类型
    const inferredMimeType = mimeType || this.inferMimeType(filename)
    formData.append('file', new Blob([file], { type: inferredMimeType }), filename)
    formData.append('document_id', documentId)

    return this.post('/process-document', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
  }

  async getProcessingStatus(documentId: string) {
    return this.get(`/processing-status/${documentId}`)
  }

  async reprocessDocument(documentId: string) {
    return this.post(`/reprocess-document/${documentId}`)
  }

  async healthCheck() {
    return this.get('/health')
  }

  // 根据文件扩展名推断MIME类型
  private inferMimeType(filename: string): string {
    const extension = filename.split('.').pop()?.toLowerCase()
    const mimeTypes: Record<string, string> = {
      'pdf': 'application/pdf',
      'doc': 'application/msword',
      'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'txt': 'text/plain',
      'md': 'text/markdown',
    }
    return mimeTypes[extension || ''] || 'application/octet-stream'
  }
}

// 向量化服务客户端
export class VectorServiceClient extends PythonServiceClient {
  constructor() {
    super(SERVICE_URLS.VECTOR_SERVICE)
  }

  async vectorizeTexts(texts: string[], storeVectors: boolean = true) {
    return this.post('/vectorize', {
      texts,
      store_vectors: storeVectors
    })
  }

  async searchVectors(queryText: string, topK: number = 5) {
    return this.post('/search', {
      query_text: queryText,
      top_k: topK
    })
  }

  async storeVectors(vectors: number[][], texts: string[], metadata?: any[]) {
    return this.post('/store-vectors', {
      vectors,
      texts,
      metadata
    })
  }

  async deleteDocumentVectors(documentId: string) {
    return this.delete(`/document/${documentId}`)
  }

  async healthCheck() {
    return this.get('/health')
  }
}

// RAG服务客户端
export class RAGServiceClient extends PythonServiceClient {
  constructor() {
    super(SERVICE_URLS.RAG_SERVICE)
  }

  async query(query: string, topK: number = 5, includeSources: boolean = true) {
    return this.post('/query', {
      query,
      top_k: topK,
      include_sources: includeSources
    })
  }

  async chat(messages: Array<{role: string, content: string}>) {
    return this.post('/chat', {
      messages
    })
  }

  async documentQA(documentId: string, question: string) {
    return this.get(`/documents/${documentId}/qa`, {
      params: { question }
    })
  }

  async healthCheck() {
    return this.get('/health')
  }
}

// 导出客户端实例
export const documentProcessor = new DocumentProcessorClient()
export const vectorService = new VectorServiceClient()
export const ragService = new RAGServiceClient()

// 保留原有函数以保持向后兼容
export async function uploadToDocumentProcessor(file: any, documentId: string) {
  return documentProcessor.uploadDocument(file, file.filename || 'unknown', documentId)
}

export async function queryRAGService(query: string) {
  return ragService.query(query)
}

export async function vectorizeTexts(texts: string[]) {
  return vectorService.vectorizeTexts(texts)
}