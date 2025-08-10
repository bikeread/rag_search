import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { documentService } from '@/services/documents'
import { toast } from 'react-hot-toast'

export const useDocuments = (params: any = {}) => {
  return useQuery({
    queryKey: ['documents', params],
    queryFn: () => documentService.getDocumentList(params),
    staleTime: 2 * 60 * 1000, // 2分钟
  })
}

export const useDocumentUpload = () => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: documentService.uploadDocument,
    onSuccess: () => {
      toast.success('文档上传成功！')
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
    onError: (error: any) => {
      toast.error(error.message || '上传失败')
    },
  })
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