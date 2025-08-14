import { NextApiRequest, NextApiResponse } from 'next'
import { prisma } from '@/lib/prisma'
import { ragService } from '@/services/pythonServices'
import { CacheService } from '@/lib/redis'
import { querySchema } from '@/lib/validation'
import { withCorsAndAuth } from '@/lib/cors'
import { AuthenticatedRequest } from '@/lib/jwtAuth'

async function handler(req: AuthenticatedRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const userId = req.user.id

    // 验证请求数据
    const validation = querySchema.safeParse(req.body)
    if (!validation.success) {
      return res.status(400).json({
        error: 'Invalid query data',
        details: validation.error.issues
      })
    }

    const { query, topK = 5, useCache = true } = validation.data

    // 生成缓存键
    const cacheKey = `query:${Buffer.from(query).toString('base64')}:${topK}`

    // 检查缓存
    if (useCache) {
      const cachedResult = await CacheService.get(cacheKey)
      if (cachedResult) {
        return res.status(200).json({
          ...cachedResult,
          cached: true
        })
      }
    }

    // 记录查询开始
    const queryRecord = await prisma.query.create({
      data: {
        text: query,
        userId: userId,
        status: 'PROCESSING',
      }
    })

    const startTime = Date.now()

    try {
      // 调用RAG服务
      const ragResult = await ragService.query(query, topK, true)
      
      const responseTime = Date.now() - startTime

      // 更新查询记录
      await prisma.query.update({
        where: { id: queryRecord.id },
        data: {
          response: ragResult.answer,
          responseTime,
          sources: ragResult.sources,
          status: 'COMPLETED',
          metadata: {
            topK,
            searchTime: ragResult.query_time,
            sourcesCount: ragResult.sources?.length || 0,
          }
        }
      })

      const result = {
        queryId: queryRecord.id,
        answer: ragResult.answer,
        sources: ragResult.sources || [],
        responseTime,
        metadata: {
          topK,
          searchTime: ragResult.query_time,
          sourcesCount: ragResult.sources?.length || 0,
        }
      }

      // 缓存结果（1小时）
      if (useCache) {
        await CacheService.set(cacheKey, result, 3600)
      }

      res.status(200).json(result)

    } catch (ragError) {
      // 更新查询记录为失败
      await prisma.query.update({
        where: { id: queryRecord.id },
        data: {
          status: 'FAILED',
          responseTime: Date.now() - startTime,
          metadata: {
            error: ragError instanceof Error ? ragError.message : 'Unknown error'
          }
        }
      })

      throw ragError
    }

  } catch (error) {
    console.error('Query error:', error)
    res.status(500).json({
      error: 'Query failed',
      details: error instanceof Error ? error.message : 'Unknown error'
    })
  }
}

export default withCorsAndAuth(handler)