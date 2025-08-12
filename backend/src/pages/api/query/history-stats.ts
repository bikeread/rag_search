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

    // 获取用户查询历史统计（排除已删除的记录）
    const total = await prisma.query.count({
      where: {
        userId: userId,
        status: {
          not: 'DELETED'
        }
      }
    })

    res.status(200).json({
      success: true,
      total
    })

  } catch (error) {
    console.error('Query history stats error:', error)
    res.status(500).json({ error: 'Failed to get history stats' })
  }
}

export default withCorsAndAuth(handler)