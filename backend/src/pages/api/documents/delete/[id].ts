import { NextApiRequest, NextApiResponse } from 'next'
import { prisma } from '@/lib/prisma'
import { CacheService } from '@/lib/redis'
import { withCorsAndAuth } from '@/lib/cors'
import { AuthenticatedRequest } from '@/lib/jwtAuth'
import { withParamValidation, globalErrorHandler, ApiError } from '@/lib/middleware'
import { deleteParamSchema } from '@/lib/validation'
import { withTransaction } from '@/lib/transaction'
import { vectorService } from '@/services/pythonServices'

async function handler(
  req: AuthenticatedRequest & { validatedQuery: { id: string } }, 
  res: NextApiResponse
) {
  if (req.method !== 'DELETE') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const { id } = req.validatedQuery
    const userId = req.user.id
    
    const result = await deleteDocumentTransaction(id, userId)
    
    // 清理缓存
    await clearUserCache(userId)
    
    res.status(200).json({
      success: true,
      message: 'Document deleted successfully',
      data: {
        documentId: id,
        deletedAt: result.updatedAt,
      }
    })
    
  } catch (error) {
    globalErrorHandler(error as Error, req, res)
  }
}

// 文档删除事务
export const deleteDocumentTransaction = async (documentId: string, userId: string) => {
  return withTransaction(async (tx) => {
    // 1. 验证文档所有权
    const document = await tx.document.findFirst({
      where: { 
        id: documentId, 
        uploadedBy: userId, 
        NOT: { status: 'DELETED' } 
      }
    })
    
    if (!document) {
      throw new ApiError(404, 'Document not found')
    }
    
    // 2. 删除向量数据库中的相关向量
    try {
      console.log(`[${new Date().toISOString()}] 开始删除文档 ${documentId} 的向量数据`)
      const vectorResult = await vectorService.deleteDocumentVectors(documentId)
      console.log(`[${new Date().toISOString()}] 向量删除结果:`, vectorResult)
    } catch (vectorError) {
      console.warn(`[${new Date().toISOString()}] 向量删除失败，继续执行文档删除:`, vectorError)
      // 不阻止文档删除流程，允许降级处理
    }
    
    // 3. 软删除文档
    const updatedDoc = await tx.document.update({
      where: { id: documentId },
      data: { 
        status: 'DELETED',
        updatedAt: new Date(),
      }
    })
    
    return updatedDoc
  })
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

// 参数预处理
function preprocessDeleteParams(query: any) {
  return {
    id: query.id as string
  }
}

export default withCorsAndAuth(
  withParamValidation(deleteParamSchema, handler)
)