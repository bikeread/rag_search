import { NextApiRequest, NextApiResponse } from 'next'
import { getServerSession } from 'next-auth/next'
import formidable from 'formidable'
import fs from 'fs/promises'
import path from 'path'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/prisma'
import { documentProcessor } from '@/services/pythonServices'
import { uploadSchema } from '@/lib/validation'

export const config = {
  api: {
    bodyParser: false,
  },
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    // 验证用户认证
    const session = await getServerSession(req, res, authOptions)
    if (!session) {
      return res.status(401).json({ error: 'Unauthorized' })
    }

    // 解析上传文件
    const form = formidable({
      multiples: false,
      maxFileSize: parseInt(process.env.MAX_FILE_SIZE || '10485760'), // 10MB
      filter: ({ mimetype }) => {
        // 允许的文件类型
        const allowedTypes = [
          'application/pdf',
          'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
          'application/msword',
          'text/plain',
          'text/markdown',
        ]
        return allowedTypes.includes(mimetype || '')
      }
    })

    const [fields, files] = await form.parse(req)
    const file = Array.isArray(files.file) ? files.file[0] : files.file

    if (!file) {
      return res.status(400).json({ error: 'No file uploaded' })
    }

    // 验证文件
    const validation = uploadSchema.safeParse({
      filename: file.originalFilename,
      size: file.size,
      mimeType: file.mimetype,
    })

    if (!validation.success) {
      return res.status(400).json({ 
        error: 'Invalid file',
        details: validation.error.issues
      })
    }

    // 创建文档记录
    const document = await prisma.document.create({
      data: {
        filename: file.originalFilename || 'unknown',
        originalName: file.originalFilename || 'unknown',
        mimeType: file.mimetype || 'application/octet-stream',
        size: file.size,
        status: 'PENDING',
        uploadedBy: (session.user as any).id,
        processingStartedAt: new Date(),
      }
    })

    // 读取文件内容
    const fileBuffer = await fs.readFile(file.filepath)

    // 发送到文档处理服务
    try {
      const processResult = await documentProcessor.uploadDocument(
        fileBuffer,
        file.originalFilename || 'unknown',
        document.id
      )

      // 更新状态为处理中
      await prisma.document.update({
        where: { id: document.id },
        data: { status: 'PROCESSING' }
      })

      res.status(200).json({
        documentId: document.id,
        status: 'processing',
        message: 'Document uploaded and processing started',
        processingInfo: processResult
      })

    } catch (processingError) {
      // 处理失败，更新状态
      await prisma.document.update({
        where: { id: document.id },
        data: {
          status: 'FAILED',
          errorMessage: processingError instanceof Error ? processingError.message : 'Processing failed'
        }
      })

      throw processingError
    }

  } catch (error) {
    console.error('Upload error:', error)
    res.status(500).json({ 
      error: 'Upload failed',
      details: error instanceof Error ? error.message : 'Unknown error'
    })
  }
}