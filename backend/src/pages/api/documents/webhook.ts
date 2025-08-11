import { NextApiRequest, NextApiResponse } from 'next'
import { prisma } from '@/lib/prisma'
import { getCorsHeaders } from '@/lib/cors'
import { CacheService } from '@/lib/redis'

// 处理Python服务的状态更新webhook
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const origin = req.headers.origin
  
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    const corsHeaders = getCorsHeaders(origin)
    Object.entries(corsHeaders).forEach(([key, value]) => {
      res.setHeader(key, value)
    })
    res.status(200).end()
    return
  }
  
  // Add CORS headers to all other requests
  const corsHeaders = getCorsHeaders(origin)
  Object.entries(corsHeaders).forEach(([key, value]) => {
    res.setHeader(key, value)
  })

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const { document_id, status, chunk_count = 0, error_message = null } = req.body

    if (!document_id || !status) {
      return res.status(400).json({ error: 'Missing required fields' })
    }

    // 验证状态
    const validStatuses = ['PENDING', 'PROCESSING', 'COMPLETED', 'FAILED']
    if (!validStatuses.includes(status)) {
      return res.status(400).json({ error: 'Invalid status' })
    }

    // 更新文档状态
    const updatedDocument = await prisma.document.update({
      where: { id: document_id },
      data: {
        status,
        errorMessage: error_message,
        ...(status === 'COMPLETED' && {
          processingCompletedAt: new Date()
        })
      },
      select: {
        id: true,
        uploadedBy: true,
        status: true
      }
    })

    // 如果有块数统计，更新chunk相关信息
    if (chunk_count > 0) {
      // 这里可以根据实际需求更新chunk数据
      console.log(`Document ${document_id} processed with ${chunk_count} chunks`)
    }

    // 清除相关缓存
    const userId = updatedDocument.uploadedBy
    await clearUserCache(userId)

    console.log(`Document ${document_id} status updated to ${status}`)
    
    res.status(200).json({
      success: true,
      message: 'Status updated successfully',
      document_id,
      status
    })

  } catch (error) {
    console.error('Webhook error:', error)
    res.status(500).json({ error: 'Internal server error' })
  }
}

// 清理用户缓存
async function clearUserCache(userId: string) {
  try {
    // 清除所有与该用户相关的文档缓存
    const patterns = [
      `documents:list:${userId}:*`,    // 列表缓存
      `documents:*:${userId}:*`,       // 其他缓存
    ]
    
    for (const pattern of patterns) {
      await CacheService.delByPattern(pattern)
    }
    
    console.log(`Cleared user cache for userId: ${userId}`)
  } catch (error) {
    console.warn('Cache clear failed:', error)
  }
}