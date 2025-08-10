import { useQuery, useMutation } from '@tanstack/react-query'
import { queryService } from '@/services/queries'
import type { QueryRequest } from '@/types'

export const useSubmitQuery = () => {
  return useMutation({
    mutationFn: (data: QueryRequest) => queryService.submitQuery(data),
    onSuccess: () => {
      // 可以在这里处理成功逻辑
    },
  })
}

export const useQueryHistory = (params: { page?: number; limit?: number } = {}) => {
  return useQuery({
    queryKey: ['queryHistory', params],
    queryFn: () => queryService.getQueryHistory(params),
    staleTime: 5 * 60 * 1000, // 5分钟
  })
}