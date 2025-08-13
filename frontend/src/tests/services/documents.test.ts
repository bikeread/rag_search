import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { documentService } from '@/services/documents'
import { apiClient } from '@/services/api'
import type { DocumentListParams, DocumentListResponse } from '@/types'

// Mock apiClient
vi.mock('@/services/api', () => ({
  apiClient: {
    get: vi.fn(),
    post: vi.fn()
  }
}))

// Mock console.error to avoid noise in test output
const mockConsoleError = vi.spyOn(console, 'error').mockImplementation(() => {})

describe('DocumentService', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    mockConsoleError.mockClear()
  })

  describe('uploadDocument', () => {
    it('should upload document successfully', async () => {
      // Arrange
      const mockFile = new File(['test content'], 'test.txt', { type: 'text/plain' })
      const mockResponse = {
        success: true,
        data: {
          documentId: 'test-doc-id',
          processingResult: {
            status: 'processing'
          }
        },
        message: 'Upload successful'
      }

      const mockApiPost = vi.mocked(apiClient.post)
      mockApiPost.mockResolvedValue(mockResponse)

      // Act
      const result = await documentService.uploadDocument(mockFile)

      // Assert
      expect(result).toEqual({
        documentId: 'test-doc-id',
        status: 'processing'
      })

      expect(mockApiPost).toHaveBeenCalledWith(
        '/api/documents/upload',
        expect.any(FormData),
        {
          headers: {
            'Content-Type': 'multipart/form-data'
          }
        }
      )

      // Verify FormData content
      const formDataCall = mockApiPost.mock.calls[0]
      const formData = formDataCall[1] as FormData
      expect(formData.get('file')).toBe(mockFile)
    })

    it('should handle upload failure', async () => {
      // Arrange
      const mockFile = new File(['test content'], 'test.txt', { type: 'text/plain' })
      const mockApiPost = vi.mocked(apiClient.post)
      mockApiPost.mockRejectedValue(new Error('Upload failed'))

      // Act & Assert
      await expect(documentService.uploadDocument(mockFile)).rejects.toThrow('文档上传失败')
      expect(console.error).toHaveBeenCalledWith('Upload failed:', expect.any(Error))
    })

    it('should handle missing status in response', async () => {
      // Arrange
      const mockFile = new File(['test content'], 'test.txt', { type: 'text/plain' })
      const mockResponse = {
        success: true,
        data: {
          documentId: 'test-doc-id',
          processingResult: {} // Missing status
        },
        message: 'Upload successful'
      }

      const mockApiPost = vi.mocked(apiClient.post)
      mockApiPost.mockResolvedValue(mockResponse)

      // Act
      const result = await documentService.uploadDocument(mockFile)

      // Assert
      expect(result).toEqual({
        documentId: 'test-doc-id',
        status: 'submitted' // Default fallback
      })
    })

    it('should create FormData with correct file', async () => {
      // Arrange
      const mockFile = new File(['test content'], 'test.pdf', { type: 'application/pdf' })
      const mockResponse = {
        success: true,
        data: {
          documentId: 'pdf-doc-id',
          processingResult: { status: 'processing' }
        },
        message: 'Upload successful'
      }

      const mockApiPost = vi.mocked(apiClient.post)
      mockApiPost.mockResolvedValue(mockResponse)

      // Act
      await documentService.uploadDocument(mockFile)

      // Assert
      expect(mockApiPost).toHaveBeenCalledTimes(1)
      const [url, formData, config] = mockApiPost.mock.calls[0]
      
      expect(url).toBe('/api/documents/upload')
      expect(formData).toBeInstanceOf(FormData)
      expect(config).toEqual({
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      })

      const uploadedFile = (formData as FormData).get('file') as File
      expect(uploadedFile).toBe(mockFile)
      expect(uploadedFile.name).toBe('test.pdf')
      expect(uploadedFile.type).toBe('application/pdf')
    })
  })

  describe('getDocuments', () => {
    it('should fetch documents successfully', async () => {
      // Arrange
      const mockParams: DocumentListParams = {
        page: 1,
        limit: 10,
        status: 'COMPLETED'
      }

      const mockResponse: DocumentListResponse = {
        documents: [
          {
            id: 'doc-1',
            filename: 'test1.txt',
            originalName: 'test1.txt',
            mimeType: 'text/plain',
            size: 100,
            status: 'COMPLETED',
            chunksCount: 2,
            createdAt: '2023-01-01T00:00:00Z',
            updatedAt: '2023-01-01T00:01:00Z'
          }
        ],
        pagination: {
          page: 1,
          limit: 10,
          total: 1,
          totalPages: 1,
          hasNext: false,
          hasPrev: false
        }
      }

      const mockApiGet = vi.mocked(apiClient.get)
      mockApiGet.mockResolvedValue(mockResponse)

      // Act
      const result = await documentService.getDocuments(mockParams)

      // Assert
      expect(result).toEqual(mockResponse)
      expect(mockApiGet).toHaveBeenCalledWith('/api/documents/list', {
        params: {
          page: 1,
          limit: 10,
          search: undefined,
          status: 'COMPLETED'
        }
      })
    })

    it('should handle fetch failure gracefully', async () => {
      // Arrange
      const mockParams: DocumentListParams = { page: 1, limit: 10 }
      const mockApiGet = vi.mocked(apiClient.get)
      mockApiGet.mockRejectedValue(new Error('Network error'))

      // Act
      const result = await documentService.getDocuments(mockParams)

      // Assert
      expect(result).toEqual({
        documents: [],
        pagination: {
          total: 0,
          page: 1,
          limit: 10,
          totalPages: 0,
          hasNext: false,
          hasPrev: false
        }
      })
      expect(console.error).toHaveBeenCalledWith('Failed to load documents:', expect.any(Error))
    })

    it('should use default params when not provided', async () => {
      // Arrange
      const mockResponse: DocumentListResponse = {
        documents: [],
        pagination: {
          page: 1,
          limit: 10,
          total: 0,
          totalPages: 0,
          hasNext: false,
          hasPrev: false
        }
      }

      const mockApiGet = vi.mocked(apiClient.get)
      mockApiGet.mockResolvedValue(mockResponse)

      // Act
      await documentService.getDocuments({})

      // Assert
      expect(mockApiGet).toHaveBeenCalledWith('/api/documents/list', {
        params: {
          page: 1,
          limit: 10,
          search: undefined,
          status: undefined
        }
      })
    })

    it('should handle search parameter', async () => {
      // Arrange
      const mockParams: DocumentListParams = {
        page: 2,
        limit: 20,
        search: 'test query',
        status: 'PROCESSING'
      }

      const mockResponse: DocumentListResponse = {
        documents: [],
        pagination: {
          page: 2,
          limit: 20,
          total: 0,
          totalPages: 0,
          hasNext: false,
          hasPrev: true
        }
      }

      const mockApiGet = vi.mocked(apiClient.get)
      mockApiGet.mockResolvedValue(mockResponse)

      // Act
      const result = await documentService.getDocuments(mockParams)

      // Assert
      expect(result).toEqual(mockResponse)
      expect(mockApiGet).toHaveBeenCalledWith('/api/documents/list', {
        params: {
          page: 2,
          limit: 20,
          search: 'test query',
          status: 'PROCESSING'
        }
      })
    })
  })

  describe('getDocumentStatus', () => {
    it('should fetch document status successfully', async () => {
      // Arrange
      const mockResponse = {
        document: {
          id: 'test-id',
          filename: 'test.txt',
          originalName: 'test.txt',
          mimeType: 'text/plain',
          size: 100,
          status: 'COMPLETED' as const,
          chunksCount: 2,
          createdAt: '2023-01-01T00:00:00Z',
          updatedAt: '2023-01-01T00:01:00Z'
        },
        processingStatus: {
          currentStep: 'completed',
          progress: 100
        },
        chunks: [
          {
            id: 'chunk-1',
            content: 'Test content',
            chunkIndex: 0,
            createdAt: '2023-01-01T00:00:30Z'
          }
        ]
      }

      const mockApiGet = vi.mocked(apiClient.get)
      mockApiGet.mockResolvedValue(mockResponse)

      // Act
      const result = await documentService.getDocumentStatus('test-id')

      // Assert
      expect(result).toEqual(mockResponse)
      expect(mockApiGet).toHaveBeenCalledWith('/api/documents/status/test-id')
    })

    it('should return default status when API fails', async () => {
      // Arrange
      const mockApiGet = vi.mocked(apiClient.get)
      mockApiGet.mockRejectedValue(new Error('API error'))

      // Act
      const result = await documentService.getDocumentStatus('test-id')

      // Assert
      expect(result).toEqual({
        document: expect.objectContaining({
          id: 'test-id',
          status: 'PENDING'
        }),
        chunks: []
      })
    })
  })

  describe('deleteDocument', () => {
    it('should return mock deletion result', async () => {
      // Act
      const result = await documentService.deleteDocument('test-id')

      // Assert
      expect(result).toEqual({
        message: '文档删除成功',
        documentId: 'test-id'
      })
    })
  })

  describe('batchDeleteDocuments', () => {
    it('should handle batch deletion', async () => {
      // Act
      const result = await documentService.batchDeleteDocuments(['id1', 'id2', 'id3'])

      // Assert
      expect(result).toEqual({
        succeeded: 3,
        failed: 0
      })
    })

    it('should handle empty array', async () => {
      // Act
      const result = await documentService.batchDeleteDocuments([])

      // Assert
      expect(result).toEqual({
        succeeded: 0,
        failed: 0
      })
    })

    it('should process documents in chunks', async () => {
      // Arrange - Create more than 5 items to test chunking
      const documentIds = Array.from({ length: 12 }, (_, i) => `doc-${i}`)

      // Act
      const result = await documentService.batchDeleteDocuments(documentIds)

      // Assert
      expect(result).toEqual({
        succeeded: 12,
        failed: 0
      })
    })
  })

  describe('edge cases', () => {
    it('should handle very large files', async () => {
      // Arrange - Create a large file (5MB)
      const largeContent = 'x'.repeat(5 * 1024 * 1024)
      const largeFile = new File([largeContent], 'large.txt', { type: 'text/plain' })
      
      const mockResponse = {
        success: true,
        data: {
          documentId: 'large-doc-id',
          processingResult: { status: 'processing' }
        },
        message: 'Upload successful'
      }

      const mockApiPost = vi.mocked(apiClient.post)
      mockApiPost.mockResolvedValue(mockResponse)

      // Act
      const result = await documentService.uploadDocument(largeFile)

      // Assert
      expect(result.documentId).toBe('large-doc-id')
      expect(result.status).toBe('processing')
    })

    it('should handle special characters in filename', async () => {
      // Arrange
      const specialFile = new File(['content'], '测试文档-特殊字符_@#$.txt', { type: 'text/plain' })
      
      const mockResponse = {
        success: true,
        data: {
          documentId: 'special-doc-id',
          processingResult: { status: 'processing' }
        },
        message: 'Upload successful'
      }

      const mockApiPost = vi.mocked(apiClient.post)
      mockApiPost.mockResolvedValue(mockResponse)

      // Act
      await documentService.uploadDocument(specialFile)

      // Assert
      const formData = mockApiPost.mock.calls[0][1] as FormData
      const uploadedFile = formData.get('file') as File
      expect(uploadedFile.name).toBe('测试文档-特殊字符_@#$.txt')
    })

    it('should handle concurrent uploads', async () => {
      // Arrange
      const files = [
        new File(['content1'], 'file1.txt', { type: 'text/plain' }),
        new File(['content2'], 'file2.txt', { type: 'text/plain' }),
        new File(['content3'], 'file3.txt', { type: 'text/plain' })
      ]

      const mockApiPost = vi.mocked(apiClient.post)
      mockApiPost
        .mockResolvedValueOnce({
          success: true,
          data: { documentId: 'doc1', processingResult: { status: 'processing' } },
          message: 'Success'
        })
        .mockResolvedValueOnce({
          success: true,
          data: { documentId: 'doc2', processingResult: { status: 'processing' } },
          message: 'Success'
        })
        .mockResolvedValueOnce({
          success: true,
          data: { documentId: 'doc3', processingResult: { status: 'processing' } },
          message: 'Success'
        })

      // Act
      const promises = files.map(file => documentService.uploadDocument(file))
      const results = await Promise.all(promises)

      // Assert
      expect(results).toHaveLength(3)
      expect(results[0].documentId).toBe('doc1')
      expect(results[1].documentId).toBe('doc2')
      expect(results[2].documentId).toBe('doc3')
      expect(mockApiPost).toHaveBeenCalledTimes(3)
    })
  })
})