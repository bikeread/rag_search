import { NextApiRequest, NextApiResponse } from 'next'
import { prisma } from '@/lib/prisma'
import { CacheService } from '@/lib/redis'
import { withCorsAndAuth } from '@/lib/cors'
import { AuthenticatedRequest } from '@/lib/jwtAuth'

async function handler(req: AuthenticatedRequest, res: NextApiResponse) {
  if (req.method !== 'DELETE') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const userId = req.user.id

    const { id } = req.query

    if (!id || typeof id !== 'string') {
      return res.status(400).json({ error: 'Invalid document ID' })
    }

    // 检查文档是否存在且属于当前用户
    const document = await prisma.document.findFirst({
      where: {
        id,
        uploadedBy: userId,
      }
    })

    if (!document) {
      return res.status(404).json({ error: 'Document not found' })
    }

    // 软删除：标记为已删除状态
    await prisma.document.update({
      where: { id },
      data: {
        status: 'DELETED',
        updatedAt: new Date(),
      }
    })

    // 清除相关缓存
    // 注意：这里需要实现模式匹配删除，简化处理可以清除用户所有文档缓存
    try {
      await CacheService.del(`documents:${userId}`)
    } catch (cacheError) {
      console.warn('Failed to clear cache:', cacheError)
    }

    res.status(200).json({
      message: 'Document deleted successfully',
      documentId: id
    })

  } catch (error) {
    console.error('Delete document error:', error)
    res.status(500).json({ error: 'Failed to delete document' })
  }
}

export default withCorsAndAuth(handler)