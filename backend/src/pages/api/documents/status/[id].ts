import { NextApiRequest, NextApiResponse } from 'next'
import { getServerSession } from 'next-auth/next'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/prisma'
import { documentProcessor } from '@/services/pythonServices'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
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

    // 从数据库获取文档信息
    const document = await prisma.document.findFirst({
      where: {
        id,
        uploadedBy: (session.user as any).id, // 确保用户只能查看自己的文档
      },
      include: {
        chunks: {
          select: {
            id: true,
            chunkIndex: true,
            createdAt: true,
          },
          orderBy: {
            chunkIndex: 'asc'
          }
        }
      }
    })

    if (!document) {
      return res.status(404).json({ error: 'Document not found' })
    }

    // 如果文档还在处理中，尝试从 Python 服务获取最新状态
    let processingStatus = null
    if (document.status === 'PROCESSING') {
      try {
        processingStatus = await documentProcessor.getProcessingStatus(document.id)
      } catch (error) {
        console.warn('Failed to get processing status from Python service:', error)
      }
    }

    res.status(200).json({
      document: {
        id: document.id,
        filename: document.filename,
        originalName: document.originalName,
        mimeType: document.mimeType,
        size: document.size,
        status: document.status,
        errorMessage: document.errorMessage,
        chunksCount: document.chunks.length,
        createdAt: document.createdAt,
        updatedAt: document.updatedAt,
        processingStartedAt: document.processingStartedAt,
        processingCompletedAt: document.processingCompletedAt,
      },
      processingStatus,
      chunks: document.chunks
    })

  } catch (error) {
    console.error('Status check error:', error)
    res.status(500).json({ error: 'Failed to get document status' })
  }
}