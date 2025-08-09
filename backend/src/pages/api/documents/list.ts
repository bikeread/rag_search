import { NextApiRequest, NextApiResponse } from 'next'
import { getServerSession } from 'next-auth/next'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/prisma'
import { CacheService } from '@/lib/redis'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const session = await getServerSession(req, res, authOptions)
    if (!session) {
      return res.status(401).json({ error: 'Unauthorized' })
    }

    const { page = '1', limit = '10', status, search } = req.query

    const pageNum = parseInt(page as string)
    const limitNum = parseInt(limit as string)
    const offset = (pageNum - 1) * limitNum

    // 构建查询条件
    const where: any = {
      uploadedBy: (session.user as any).id,
    }

    if (status && typeof status === 'string') {
      where.status = status.toUpperCase()
    }

    if (search && typeof search === 'string') {
      where.OR = [
        { filename: { contains: search, mode: 'insensitive' } },
        { originalName: { contains: search, mode: 'insensitive' } },
      ]
    }

    // 检查缓存
    const cacheKey = `documents:${(session.user as any).id}:${pageNum}:${limitNum}:${status || 'all'}:${search || ''}`
    const cachedResult = await CacheService.get(cacheKey)

    if (cachedResult) {
      return res.status(200).json(cachedResult)
    }

    // 查询文档
    const [documents, total] = await Promise.all([
      prisma.document.findMany({
        where,
        include: {
          _count: {
            select: {
              chunks: true
            }
          }
        },
        orderBy: {
          createdAt: 'desc'
        },
        skip: offset,
        take: limitNum,
      }),
      prisma.document.count({ where })
    ])

    const result = {
      documents: documents.map(doc => ({
        id: doc.id,
        filename: doc.filename,
        originalName: doc.originalName,
        mimeType: doc.mimeType,
        size: doc.size,
        status: doc.status,
        chunksCount: doc._count.chunks,
        createdAt: doc.createdAt,
        updatedAt: doc.updatedAt,
        processingStartedAt: doc.processingStartedAt,
        processingCompletedAt: doc.processingCompletedAt,
      })),
      pagination: {
        page: pageNum,
        limit: limitNum,
        total,
        totalPages: Math.ceil(total / limitNum),
        hasNext: pageNum * limitNum < total,
        hasPrev: pageNum > 1,
      }
    }

    // 缓存结果（5分钟）
    await CacheService.set(cacheKey, result, 300)

    res.status(200).json(result)

  } catch (error) {
    console.error('List documents error:', error)
    res.status(500).json({ error: 'Failed to list documents' })
  }
}