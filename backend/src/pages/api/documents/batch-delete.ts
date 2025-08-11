import { NextApiRequest, NextApiResponse } from 'next'
import { prisma } from '@/lib/prisma'
import { CacheService } from '@/lib/redis'
import { withCorsAndAuth } from '@/lib/cors'
import { AuthenticatedRequest } from '@/lib/jwtAuth'
import { withTransaction } from '@/lib/transaction'
import { globalErrorHandler, ApiError } from '@/lib/middleware'
import { vectorService } from '@/services/pythonServices'

interface BatchDeleteRequest {
  documentIds: string[]
}

interface BatchDeleteResult {
  success: string[]
  failed: Array<{
    documentId: string
    error: string
  }>
  total: number
}

async function handler(req: AuthenticatedRequest, res: NextApiResponse) {
  if (req.method !== 'DELETE') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const userId = req.user.id
    const { documentIds }: BatchDeleteRequest = req.body
    
    if (!documentIds || !Array.isArray(documentIds) || documentIds.length === 0) {
      throw new ApiError(400, 'Document IDs array is required')
    }
    
    if (documentIds.length > 20) {
      throw new ApiError(400, 'Maximum 20 documents can be deleted at once')
    }
    
    const results: BatchDeleteResult = {
      success: [],
      failed: [],
      total: documentIds.length
    }
    
    // 1. 验证所有文档的所有权
    const documents = await prisma.document.findMany({
      where: {
        id: { in: documentIds },
        uploadedBy: userId,
        NOT: { status: 'DELETED' }
      },
      select: { id: true, originalName: true }
    })
    
    const validDocIds = documents.map(doc => doc.id)
    const docNameMap = documents.reduce((map, doc) => {
      map[doc.id] = doc.originalName
      return map
    }, {} as Record<string, string>)
    
    // 标记无效的文档ID
    documentIds.forEach(id => {
      if (!validDocIds.includes(id)) {
        results.failed.push({
          documentId: id,
          error: 'Document not found or already deleted'
        })
      }
    })
    
    // 2. 批量删除有效文档
    if (validDocIds.length > 0) {
      // 并行处理，每批5个
      const batchSize = 5
      for (let i = 0; i < validDocIds.length; i += batchSize) {
        const batch = validDocIds.slice(i, i + batchSize)
        
        await Promise.allSettled(
          batch.map(async (documentId) => {
            try {
              await withTransaction(async (tx) => {
                // 删除向量数据库中的向量
                try {
                  console.log(`[${new Date().toISOString()}] 删除文档 ${documentId} 的向量数据`)
                  const vectorResult = await vectorService.deleteDocumentVectors(documentId)
                  console.log(`[${new Date().toISOString()}] 向量删除结果:`, vectorResult)
                } catch (vectorError) {
                  console.warn(`[${new Date().toISOString()}] 向量删除失败，继续执行文档删除:`, vectorError)
                  // 不阻止文档删除流程
                }
                
                // 软删除文档
                await tx.document.update({
                  where: { id: documentId },
                  data: { 
                    status: 'DELETED',
                    updatedAt: new Date(),
                  }
                })
              })
              
              results.success.push(documentId)
              console.log(`[${new Date().toISOString()}] 成功删除文档: ${docNameMap[documentId]} (${documentId})`)
              
            } catch (error) {
              results.failed.push({
                documentId: documentId,
                error: error.message || 'Unknown error occurred'
              })
              console.error(`[${new Date().toISOString()}] 删除文档失败 ${documentId}:`, error)
            }
          })
        )
      }
    }
    
    // 3. 清理用户缓存
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
      message: `Batch delete completed: ${results.success.length} succeeded, ${results.failed.length} failed`,
      data: results
    })
    
  } catch (error) {
    console.error('Batch delete error:', error)
    globalErrorHandler(error as Error, req, res)
  }
}

export default withCorsAndAuth(handler)