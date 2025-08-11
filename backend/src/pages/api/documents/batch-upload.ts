import { NextApiRequest, NextApiResponse } from 'next'
import formidable from 'formidable'
import fs from 'fs/promises'
import path from 'path'
import { prisma } from '@/lib/prisma'
import { documentProcessor } from '@/services/pythonServices'
import { validateFileType } from '@/lib/validation'
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

interface BatchUploadResult {
  success: string[]
  failed: Array<{
    filename: string
    error: string
  }>
  total: number
}

async function handler(req: AuthenticatedRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const userId = req.user.id
    const results: BatchUploadResult = {
      success: [],
      failed: [],
      total: 0
    }
    
    // 1. 解析多个文件
    const form = formidable({
      multiples: true,
      maxFiles: 10,  // 最多10个文件
      maxFileSize: 50 * 1024 * 1024, // 50MB per file
      filter: function ({ mimetype }) {
        return validateFileType(mimetype || '')
      }
    })
    
    const [fields, files] = await form.parse(req)
    
    // 确保files.file是数组
    const fileArray = Array.isArray(files.file) ? files.file : [files.file].filter(Boolean)
    
    if (fileArray.length === 0) {
      throw new ApiError(400, 'No valid files provided')
    }
    
    results.total = fileArray.length
    
    // 2. 并行处理所有文件（限制并发数）
    const batchSize = 3 // 同时处理3个文件
    for (let i = 0; i < fileArray.length; i += batchSize) {
      const batch = fileArray.slice(i, i + batchSize)
      
      await Promise.allSettled(
        batch.map(async (file) => {
          if (!file) return
          
          try {
            // 验证文件
            if (!validateFileType(file.mimetype || '')) {
              throw new Error(`Unsupported file type: ${file.mimetype}`)
            }
            
            if (file.size > 50 * 1024 * 1024) {
              throw new Error(`File too large: ${file.size} bytes`)
            }
            
            // 创建文档记录
            const document = await withTransaction(async (tx) => {
              return tx.document.create({
                data: {
                  filename: file.originalFilename || file.newFilename || 'unknown',
                  originalName: file.originalFilename || file.newFilename || 'unknown',
                  mimeType: file.mimetype || 'application/octet-stream',
                  size: file.size,
                  uploadedBy: userId,
                  status: 'PENDING',
                  processingStartedAt: new Date(),
                }
              })
            })
            
            // 读取文件内容
            const fileBuffer = await fs.readFile(file.filepath)
            
            // 发送给Document Processor
            try {
              await documentProcessor.uploadDocument(
                fileBuffer,
                document.filename,
                document.id,
                document.mimeType
              )
              
              // 更新状态为处理中
              await prisma.document.update({
                where: { id: document.id },
                data: { status: 'PROCESSING' }
              })
              
              results.success.push(document.originalName)
              console.log(`[${new Date().toISOString()}] 批量上传成功: ${document.originalName}`)
              
            } catch (processingError) {
              // 处理失败，标记文档状态
              await prisma.document.update({
                where: { id: document.id },
                data: { 
                  status: 'FAILED',
                  errorMessage: `处理失败: ${processingError.message}`
                }
              })
              
              results.failed.push({
                filename: document.originalName,
                error: `Processing failed: ${processingError.message}`
              })
            }
            
          } catch (fileError) {
            results.failed.push({
              filename: file.originalFilename || file.newFilename || 'unknown',
              error: fileError.message
            })
            console.error(`[${new Date().toISOString()}] 文件处理失败:`, fileError)
          } finally {
            // 清理临时文件
            try {
              await fs.unlink(file.filepath)
            } catch (cleanupError) {
              console.warn('Failed to cleanup temp file:', cleanupError)
            }
          }
        })
      )
    }
    
    // 3. 清理缓存
    try {
      const pattern = `documents:*:${userId}:*`
      const keys = await CacheService.keys ? await CacheService.keys(pattern) : []
      if (keys.length > 0) {
        await CacheService.del(keys)
      }
    } catch (cacheError) {
      console.warn('Cache clear failed:', cacheError)
    }
    
    // 4. 返回结果
    const statusCode = results.failed.length === 0 ? 200 : 207 // 207 Multi-Status
    
    res.status(statusCode).json({
      success: true,
      message: `Batch upload completed: ${results.success.length} succeeded, ${results.failed.length} failed`,
      data: results
    })
    
  } catch (error) {
    console.error('Batch upload error:', error)
    globalErrorHandler(error as Error, req, res)
  }
}

export default withCorsAndAuth(handler)