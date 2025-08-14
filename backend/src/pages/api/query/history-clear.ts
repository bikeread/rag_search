import { NextApiRequest, NextApiResponse } from 'next'
import { prisma } from '@/lib/prisma'
import { withCorsAndAuth } from '@/lib/cors'
import { AuthenticatedRequest } from '@/lib/jwtAuth'

async function handler(req: AuthenticatedRequest, res: NextApiResponse) {
  if (req.method !== 'DELETE') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const userId = req.user.id
    const { confirm } = req.body

    // 确认参数验证
    if (confirm !== true) {
      return res.status(400).json({ 
        error: 'Missing confirmation parameter' 
      })
    }

    // 软删除：更新status为DELETED而不是物理删除
    const result = await prisma.query.updateMany({
      where: {
        userId: userId,
        status: {
          not: 'DELETED' // 避免重复删除
        }
      },
      data: {
        status: 'DELETED',
        deletedAt: new Date()
      }
    })

    res.status(200).json({
      success: true,
      message: 'Query history cleared successfully',
      deletedCount: result.count
    })

  } catch (error) {
    console.error('Clear history error:', error)
    res.status(500).json({ error: 'Failed to clear history' })
  }
}

export default withCorsAndAuth(handler)