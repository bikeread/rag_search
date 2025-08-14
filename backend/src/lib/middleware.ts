import { NextApiRequest, NextApiResponse } from 'next'
import { z } from 'zod'
import { AuthenticatedRequest } from './jwtAuth'

export const withParamValidation = <T extends z.ZodSchema>(
  schema: T,
  handler: (
    req: AuthenticatedRequest & { validatedQuery: z.infer<T> }, 
    res: NextApiResponse
  ) => Promise<void>
) => {
  return async (req: AuthenticatedRequest, res: NextApiResponse) => {
    try {
      // 参数预处理和类型转换
      const processedQuery = preprocessQueryParams(req.query)
      
      // Schema验证
      const validatedQuery = schema.parse(processedQuery)
      
      // 注入验证后的参数
      const enhancedReq = req as AuthenticatedRequest & { validatedQuery: z.infer<T> }
      enhancedReq.validatedQuery = validatedQuery
      
      return handler(enhancedReq, res)
    } catch (error) {
      if (error instanceof z.ZodError) {
        return res.status(400).json({
          success: false,
          error: 'Invalid parameters',
          details: error.issues
        })
      }
      
      return res.status(400).json({
        success: false,
        error: 'Parameter validation failed',
        details: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }
}

// 参数预处理函数
function preprocessQueryParams(query: any) {
  const processed: any = {}
  
  Object.keys(query).forEach(key => {
    const value = query[key]
    
    if (value === undefined || value === null) {
      return
    }
    
    // ID参数直接传递
    if (key === 'id') {
      processed[key] = value as string
      return
    }
    
    // 数字类型转换
    if (key === 'page' || key === 'limit') {
      processed[key] = parseInt(value as string) || undefined
      return
    }
    
    // 状态值大写转换
    if (key === 'status' && typeof value === 'string') {
      processed[key] = value.toUpperCase()
      return
    }
    
    // 搜索字符串处理
    if (key === 'search' && typeof value === 'string') {
      const trimmed = value.trim()
      processed[key] = trimmed.length > 0 ? trimmed : undefined
      return
    }
    
    processed[key] = value
  })
  
  return processed
}

// 自定义API错误类
export class ApiError extends Error {
  constructor(
    public statusCode: number,
    message: string,
    public code?: string,
    public details?: any
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

// 全局错误处理器
export const globalErrorHandler = (
  error: Error, 
  req: NextApiRequest, 
  res: NextApiResponse
) => {
  console.error('API Error:', {
    error: error.message,
    stack: error.stack,
    url: req.url,
    method: req.method,
    userId: (req as any).user?.id,
    timestamp: new Date().toISOString(),
  })

  // 错误分类和响应
  if (error instanceof ApiError) {
    return res.status(error.statusCode).json({
      success: false,
      error: error.message,
      code: error.code,
      details: error.details,
      timestamp: new Date().toISOString(),
    })
  }

  if (error.name === 'PrismaClientKnownRequestError') {
    return res.status(500).json({
      success: false,
      error: 'Database operation failed',
      code: 'DATABASE_ERROR',
      timestamp: new Date().toISOString(),
    })
  }

  // 默认错误响应
  return res.status(500).json({
    success: false,
    error: 'Internal server error',
    code: 'INTERNAL_ERROR',
    timestamp: new Date().toISOString(),
  })
}