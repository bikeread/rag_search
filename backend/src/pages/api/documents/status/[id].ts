import { NextApiRequest, NextApiResponse } from 'next'
import { prisma } from '@/lib/prisma'
import { documentProcessor } from '@/services/pythonServices'
import { withCorsAndAuth } from '@/lib/cors'
import { AuthenticatedRequest } from '@/lib/jwtAuth'
import { withParamValidation, globalErrorHandler, ApiError } from '@/lib/middleware'
import { statusParamSchema } from '@/lib/validation'

async function handler(
  req: AuthenticatedRequest & { validatedQuery: { id: string } }, 
  res: NextApiResponse
) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const { id } = req.validatedQuery
    const userId = req.user.id
    
    // 获取数据库状态
    const document = await getDocumentWithChunks(id, userId)
    
    if (!document) {
      return res.status(404).json({
        success: false,
        error: 'Document not found'
      })
    }
    
    // 处理中文档获取Python服务状态
    let processingStatus = null
    if (document.status === 'PROCESSING') {
      processingStatus = await documentProcessor
        .getProcessingStatus(id)
        .catch(error => {
          console.warn('Python service status unavailable:', error)
          return null
        })
    }
    
    res.status(200).json({
      success: true,
      data: {
        document: formatDocumentResponse(document),
        processingStatus,
        chunks: formatChunksResponse(document.chunks),
      }
    })
    
  } catch (error) {
    globalErrorHandler(error as Error, req, res)
  }
}

// 获取文档及其块信息
async function getDocumentWithChunks(documentId: string, userId: string) {
  return await prisma.document.findFirst({
    where: { 
      id: documentId, 
      uploadedBy: userId,
      NOT: { status: 'DELETED' }
    },
    include: {
      chunks: {
        select: {
          id: true,
          chunkIndex: true,
          content: true,
          metadata: true,
          createdAt: true,
        },
        orderBy: { chunkIndex: 'asc' },
      },
    },
  })
}

// 格式化文档响应
function formatDocumentResponse(doc: any) {
  return {
    id: doc.id,
    filename: doc.filename,
    originalName: doc.originalName,
    mimeType: doc.mimeType,
    size: doc.size,
    status: doc.status,
    chunksCount: doc.chunks?.length || 0,
    createdAt: doc.createdAt.toISOString(),
    updatedAt: doc.updatedAt.toISOString(),
    processingStartedAt: doc.processingStartedAt?.toISOString(),
    processingCompletedAt: doc.processingCompletedAt?.toISOString(),
    errorMessage: doc.errorMessage,
  }
}

// 格式化块响应
function formatChunksResponse(chunks: any[]) {
  return chunks.map(chunk => ({
    id: chunk.id,
    chunkIndex: chunk.chunkIndex,
    content: chunk.content,
    metadata: chunk.metadata,
    createdAt: chunk.createdAt.toISOString(),
  }))
}

export default withCorsAndAuth(
  withParamValidation(statusParamSchema, handler)
)