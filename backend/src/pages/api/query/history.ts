import { NextApiRequest, NextApiResponse } from 'next'
import jwt from 'jsonwebtoken'
import { prisma } from '@/lib/prisma'
import { withCors } from '@/lib/cors'

async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    // JWT验证 - 与query API保持一致
    const authorization = req.headers.authorization
    if (!authorization || !authorization.startsWith('Bearer ')) {
      return res.status(401).json({ error: 'No token provided' })
    }

    const token = authorization.substring(7)
    const JWT_SECRET = process.env.JWT_SECRET || 'fallback-secret-key'
    
    const decoded = jwt.verify(token, JWT_SECRET) as any
    const userId = decoded.userId

    const { page = '1', limit = '20', status } = req.query

    const pageNum = parseInt(page as string)
    const limitNum = parseInt(limit as string)
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

export default withCors(handler)