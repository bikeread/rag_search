import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import React, { useState, useCallback } from 'react'
import { documentService } from '@/services/documents'
import { toast } from 'react-hot-toast'
import type { DocumentListParams, Document } from '@/types'

export const useDocuments = (params: DocumentListParams = {}) => {
  return useQuery({
    queryKey: ['documents', params],
    queryFn: () => documentService.getDocuments(params),
    staleTime: 2 * 60 * 1000, // 2分钟
    retry: (failureCount, error: any) => {
      if (error?.message?.includes('401')) return false
      return failureCount < 3
    },
    retryDelay: attemptIndex => Math.min(1000 * 2 ** attemptIndex, 30000),
  })
}

export const useDocumentUpload = () => {
  const queryClient = useQueryClient()
  const [uploadProgress, setUploadProgress] = useState<{
    [fileId: string]: {
      progress: number
      status: 'uploading' | 'processing' | 'completed' | 'failed'
      phase: string
    }
  }>({})

  const generateFileId = (file: File) => {
    return `${file.name}-${file.size}-${Date.now()}`
  }

  const mutation = useMutation({
    mutationFn: async (file: File) => {
      const fileId = generateFileId(file)
      
      setUploadProgress(prev => ({
        ...prev,
        [fileId]: { progress: 0, status: 'uploading', phase: '准备上传...' }
      }))

      const result = await documentService.uploadDocument(file, (progress) => {
        setUploadProgress(prev => ({
          ...prev,
          [fileId]: { 
            progress, 
            status: 'uploading', 
            phase: `上传中... ${progress}%` 
          }
        }))
      })

      setUploadProgress(prev => ({
        ...prev,
        [fileId]: { progress: 100, status: 'processing', phase: '文档处理中...' }
      }))

      return { ...result, fileId }
    },
    onSuccess: (data) => {
      toast.success('文档上传成功！')
      queryClient.invalidateQueries({ queryKey: ['documents'] })
      
      // 清理进度状态
      setTimeout(() => {
        setUploadProgress(prev => {
          const { [data.fileId]: _removed, ...rest } = prev
          return rest
        })
      }, 3000)
    },
    onError: (error: any, file) => {
      const fileId = generateFileId(file)
      toast.error(error.message || '上传失败')
      
      setUploadProgress(prev => ({
        ...prev,
        [fileId]: { progress: 0, status: 'failed', phase: '上传失败' }
      }))
    },
  })

  return {
    ...mutation,
    uploadProgress,
  }
}

export const useDocumentDelete = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: documentService.deleteDocument,
    onSuccess: () => {
      toast.success('文档删除成功！')
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
    onError: (error: any) => {
      toast.error(error.message || '删除失败')
    },
  })
}

export const useBatchOperations = () => {
  const [selectedRows, setSelectedRows] = useState<string[]>([])
  const [batchLoading, setBatchLoading] = useState(false)
  const queryClient = useQueryClient()

  const batchDelete = useMutation({
    mutationFn: async (documentIds: string[]) => {
      setBatchLoading(true)
      return documentService.batchDeleteDocuments(documentIds)
    },
    onSuccess: ({ succeeded, failed }) => {
      toast.success(
        `成功删除 ${succeeded} 个文档${failed > 0 ? `，${failed} 个失败` : ''}`
      )
      setSelectedRows([])
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
    onError: (error: any) => {
      toast.error(error.message || '批量删除失败')
    },
    onSettled: () => {
      setBatchLoading(false)
    },
  })

  const rowSelection = {
    selectedRowKeys: selectedRows,
    onChange: (selectedRowKeys: React.Key[]) => {
      setSelectedRows(selectedRowKeys as string[])
    },
    getCheckboxProps: (record: Document) => ({
      disabled: record.status === 'PROCESSING',
    }),
  }

  return {
    selectedRows,
    batchLoading,
    rowSelection,
    batchDelete: batchDelete.mutate,
    clearSelection: () => setSelectedRows([]),
  }
}

export const useResponsiveView = () => {
  const [isMobile, setIsMobile] = useState(false)
  
  const checkMobile = useCallback(() => {
    setIsMobile(window.innerWidth < 768)
  }, [])
  
  React.useEffect(() => {
    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [checkMobile])
  
  return { isMobile }
}