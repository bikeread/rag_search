import { z } from 'zod'

export const uploadSchema = z.object({
  filename: z.string().min(1).max(255),
  size: z.number().min(1).max(10 * 1024 * 1024), // 10MB
  mimeType: z.string(),
})

export const querySchema = z.object({
  query: z.string().min(1).max(1000),
  topK: z.number().min(1).max(20).optional(),
  useCache: z.boolean().optional(),
})

export const chatSchema = z.object({
  messages: z.array(z.object({
    role: z.enum(['user', 'assistant']),
    content: z.string().min(1),
  })).min(1).max(50),
})

export const registerSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8).max(128),
  name: z.string().min(1).max(100).optional(),
})

// 文档列表查询参数验证
export const listQuerySchema = z.object({
  page: z.number().min(1).default(1),
  limit: z.number().min(1).max(100).default(10),
  status: z.enum(['PENDING', 'PROCESSING', 'COMPLETED', 'FAILED']).optional(),
  search: z.string().optional(),
})

// 文档删除参数验证
export const deleteParamSchema = z.object({
  id: z.string().cuid('Invalid document ID'),
})

// 文档状态查询参数验证
export const statusParamSchema = z.object({
  id: z.string().cuid('Invalid document ID'),
})

// 增强的文件上传验证
export const enhancedUploadSchema = z.object({
  filename: z.string()
    .min(1, '文件名不能为空')
    .max(255, '文件名过长')
    .regex(/^[a-zA-Z0-9\u4e00-\u9fa5._-]+$/, '文件名包含非法字符'),
  size: z.number()
    .positive('文件大小必须大于0')
    .max(50 * 1024 * 1024, '文件大小超过限制'),
  mimeType: z.string().min(1, 'MIME类型不能为空'),
})

// 文件类型验证函数（单独处理，支持.md文件的特殊情况）
export function validateFileType(filename: string, mimeType: string): boolean {
  const allowedTypes = [
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/msword',
    'text/plain',
    'text/markdown',
    'text/x-markdown',
  ]
  
  // 检查MIME类型
  if (allowedTypes.includes(mimeType)) {
    return true
  }
  
  // 对于.md文件，允许text/plain类型（系统通常识别为text/plain）
  if (filename.toLowerCase().endsWith('.md') && 
      (mimeType === 'text/plain' || mimeType === 'application/octet-stream')) {
    return true
  }
  
  return false
}