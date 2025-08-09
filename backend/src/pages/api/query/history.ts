import { NextApiRequest, NextApiResponse } from 'next'
import { getServerSession } from 'next-auth/next'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/prisma'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const session = await getServerSession(req, res, authOptions)
    if (!session) {
      return res.status(401).json({ error: 'Unauthorized' })
    }

    const { page = '1', limit = '20', status } = req.query

    const pageNum = parseInt(page as string)
    const limitNum = parseInt(limit as string)
    const offset = (pageNum - 1) * limitNum

    const where: any = {
      userId: (session.user as any).id,
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