import { NextApiRequest, NextApiResponse } from 'next'
import { prisma } from '@/lib/prisma'
import { withCorsAndAuth } from '@/lib/cors'
import { AuthenticatedRequest } from '@/lib/jwtAuth'

async function handler(req: AuthenticatedRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const userId = req.user.id

    const { page = '1', limit = '20', status } = req.query

    // 参数验证和默认值处理
    const pageNum = Math.max(1, parseInt(page as string) || 1)
    const limitNum = Math.min(100, Math.max(1, parseInt(limit as string) || 20)) // 限制最大100条
    const offset = (pageNum - 1) * limitNum

    const where: any = {
      userId: userId,
    }

    if (status && typeof status === 'string') {
      where.status = status.toUpperCase()
    }

    const [queries, total] = await Promise.all([
      prisma.query.findMany({
        where,
        orderBy: {
          createdAt: 'desc'
        },
        skip: offset,
        take: limitNum,
        select: {
          id: true,
          text: true,
          response: true,
          responseTime: true,
          status: true,
          createdAt: true,
          metadata: true,
        }
      }),
      prisma.query.count({ where })
    ])

    res.status(200).json({
      queries,
      pagination: {
        page: pageNum,
        limit: limitNum,
        total,
        totalPages: Math.ceil(total / limitNum),
        hasNext: pageNum * limitNum < total,
        hasPrev: pageNum > 1,
      }
    })

  } catch (error) {
    console.error('Query history error:', error)
    res.status(500).json({ error: 'Failed to get query history' })
  }
}

export default withCorsAndAuth(handler)