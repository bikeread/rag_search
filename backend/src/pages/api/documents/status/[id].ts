import { NextApiRequest, NextApiResponse } from 'next'
import { prisma } from '@/lib/prisma'
import { documentProcessor } from '@/services/pythonServices'
import { withCorsAndAuth } from '@/lib/cors'
import { AuthenticatedRequest } from '@/lib/jwtAuth'

async function handler(req: AuthenticatedRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const userId = req.user.id

    const { id } = req.query

    if (!id || typeof id !== 'string') {
      return res.status(400).json({ error: 'Invalid document ID' })
    }

    // 从数据库获取文档信息
    const document = await prisma.document.findFirst({
      where: {
        id,
        uploadedBy: userId, // 确保用户只能查看自己的文档
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

export default withCorsAndAuth(handler)