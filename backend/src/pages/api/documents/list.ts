import { NextApiRequest, NextApiResponse } from 'next'
import { prisma } from '@/lib/prisma'
import { CacheService } from '@/lib/redis'
import { withCorsAndAuth } from '@/lib/cors'
import { AuthenticatedRequest } from '@/lib/jwtAuth'
import { withParamValidation, globalErrorHandler } from '@/lib/middleware'
import { listQuerySchema } from '@/lib/validation'
import { createHash } from 'crypto'

async function handler(
  req: AuthenticatedRequest & { validatedQuery: { page: number; limit: number; status?: string; search?: string } }, 
  res: NextApiResponse
) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const { page, limit, status, search } = req.validatedQuery
    const userId = req.user.id
    const offset = (page - 1) * limit

    // 智能缓存键生成
    const cacheKey = generateCacheKey('list', userId, req.validatedQuery)
    
    // 尝试缓存命中
    const cached = await CacheService.get(cacheKey)
    if (cached) {
      return res.status(200).json(cached)
    }

    // 构建查询条件
    const where: any = {
      uploadedBy: userId,
      NOT: { status: 'DELETED' }, // 排除软删除
    }

    if (status) {
      where.status = status
    }

    if (search) {
      where.OR = [
        { originalName: { contains: search, mode: 'insensitive' } },
        { filename: { contains: search, mode: 'insensitive' } },
      ]
    }

    // 并行查询数据和总数
    const [documents, total] = await Promise.all([
      prisma.document.findMany({
        where,
        select: {
          id: true,
          filename: true,
          originalName: true,
          mimeType: true,
          size: true,
          status: true,
          createdAt: true,
          updatedAt: true,
          processingStartedAt: true,
          processingCompletedAt: true,
          errorMessage: true,
          _count: {
            select: {
              chunks: true
            }
          }
        },
        orderBy: { createdAt: 'desc' },
        skip: offset,
        take: limit,
      }),
      prisma.document.count({ where }),
    ])

    const result = {
      success: true,
      documents: documents.map(formatDocumentResponse),
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
        hasNext: page * limit < total,
        hasPrev: page > 1,
      },
    }

    // 缓存结果（5分钟）
    await CacheService.set(cacheKey, result, 300)
    res.status(200).json(result)

  } catch (error) {
    globalErrorHandler(error as Error, req, res)
  }
}

// 缓存键生成函数
function generateCacheKey(type: string, userId: string, params: any): string {
  const sortedParams = Object.keys(params)
    .sort()
    .reduce((result, key) => {
      result[key] = params[key]
      return result
    }, {} as any)
  
  const paramHash = createHash('md5')
    .update(JSON.stringify(sortedParams))
    .digest('hex')
  
  return `documents:${type}:${userId}:${paramHash}`
}

// 格式化文档响应
function formatDocumentResponse(doc: any) {
  return {
    id: doc.id,
    filename: doc.filename,
    originalName: doc.originalName,
    mimeType: doc.mimeType,
    size: doc.size,
    status: doc.status,
    chunksCount: doc._count?.chunks || 0,
    createdAt: doc.createdAt.toISOString(),
    updatedAt: doc.updatedAt.toISOString(),
    processingStartedAt: doc.processingStartedAt?.toISOString(),
    processingCompletedAt: doc.processingCompletedAt?.toISOString(),
    errorMessage: doc.errorMessage,
  }
}

export default withCorsAndAuth(
  withParamValidation(listQuerySchema, handler)
)