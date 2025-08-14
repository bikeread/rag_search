import { NextApiRequest, NextApiResponse } from 'next'
import formidable from 'formidable'
import fs from 'fs/promises'
import path from 'path'
import { prisma } from '@/lib/prisma'
import { documentProcessor } from '@/services/pythonServices'
import { enhancedUploadSchema, validateFileType } from '@/lib/validation'
import { withCorsAndAuth } from '@/lib/cors'
import { AuthenticatedRequest } from '@/lib/jwtAuth'
import { withTransaction } from '@/lib/transaction'
import { globalErrorHandler, ApiError } from '@/lib/middleware'
import { CacheService } from '@/lib/redis'

export const config = {
  api: {
    bodyParser: false,
  },
}

async function handler(req: AuthenticatedRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const userId = req.user.id
    
    const result = await withTransaction(async (tx) => {
      // 1. 文件解析和验证
      const { file, validation } = await parseAndValidateFile(req)
      
      // 2. 创建文档记录
      const document = await tx.document.create({
        data: {
          filename: validation.data.filename,
          originalName: validation.data.filename,
          mimeType: validation.data.mimeType,
          size: validation.data.size,
          uploadedBy: userId,
          status: 'PENDING',
          processingStartedAt: new Date(),
        }
      })
      
      // 3. 读取文件内容
      const fileBuffer = await fs.readFile(file.filepath)
      
      // 4. 异步处理
      const processingResult = await documentProcessor.uploadDocument(
        fileBuffer,
        file.originalFilename || 'unknown',
        document.id,
        validation.data.mimeType
      ).catch(error => {
        // Python服务失败，标记为失败
        tx.document.update({
          where: { id: document.id },
          data: { 
            status: 'FAILED',
            errorMessage: error.message 
          }
        })
        throw error
      })
      
      // 5. 状态更新
      await tx.document.update({
        where: { id: document.id },
        data: { status: 'PROCESSING' }
      })
      
      return { documentId: document.id, processingResult }
    })
    
    // 6. 缓存失效
    await clearUserCache(userId)
    
    res.status(201).json({
      success: true,
      data: result,
      message: 'Document uploaded successfully'
    })
    
  } catch (error) {
    globalErrorHandler(error as Error, req, res)
  }
}

// 文件解析和验证
async function parseAndValidateFile(req: NextApiRequest) {
  const form = formidable({
    maxFileSize: 50 * 1024 * 1024, // 50MB
    keepExtensions: true,
    filter: ({ mimetype, originalFilename }) => {
      const allowedTypes = [
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword',
        'text/plain',
        'text/markdown',
        'text/x-markdown', // 另一种Markdown MIME类型
      ]
      
      // 检查MIME类型
      if (allowedTypes.includes(mimetype || '')) {
        return true
      }
      
      // 对于.md文件，基于文件扩展名判断（因为MIME类型可能不准确）
      if (originalFilename && originalFilename.toLowerCase().endsWith('.md')) {
        return true
      }
      
      return false
    }
  })
  
  const [fields, files] = await form.parse(req)
  const file = Array.isArray(files.file) ? files.file[0] : files.file
  
  if (!file) {
    throw new ApiError(400, 'No file provided')
  }
  
  // 文件验证
  const validation = enhancedUploadSchema.safeParse({
    filename: file.originalFilename,
    size: file.size,
    mimeType: file.mimetype,
  })
  
  if (!validation.success) {
    throw new ApiError(400, 'Invalid file', 'VALIDATION_ERROR', validation.error.issues)
  }
  
  // 额外的文件类型验证（支持.md文件特殊情况）
  if (!validateFileType(validation.data.filename, validation.data.mimeType)) {
    throw new ApiError(400, 'Invalid file', 'VALIDATION_ERROR', [
      { code: 'custom', path: ['mimeType'], message: '不支持的文件类型' }
    ])
  }
  
  return { file, validation }
}

// 清理用户缓存
async function clearUserCache(userId: string) {
  try {
    const pattern = `documents:*:${userId}:*`
    const keys = await CacheService.keys ? await CacheService.keys(pattern) : []
    if (keys.length > 0) {
      await CacheService.del(keys)
    }
  } catch (error) {
    console.warn('Cache clear failed:', error)
  }
}

export default withCorsAndAuth(handler)