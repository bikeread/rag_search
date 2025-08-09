import { NextApiRequest, NextApiResponse } from 'next'
import { getServerSession } from 'next-auth/next'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/prisma'
import { CacheService } from '@/lib/redis'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'DELETE') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const session = await getServerSession(req, res, authOptions)
    if (!session) {
      return res.status(401).json({ error: 'Unauthorized' })
    }

    const { id } = req.query

    if (!id || typeof id !== 'string') {
      return res.status(400).json({ error: 'Invalid document ID' })
    }

    // 检查文档是否存在且属于当前用户
    const document = await prisma.document.findFirst({
      where: {
        id,
        uploadedBy: (session.user as any).id,
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
      await CacheService.del(`documents:${(session.user as any).id}`)
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