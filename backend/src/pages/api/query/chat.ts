import { NextApiRequest, NextApiResponse } from 'next'
import { getServerSession } from 'next-auth/next'
import { authOptions } from '@/lib/auth'
import { ragService } from '@/services/pythonServices'
import { chatSchema } from '@/lib/validation'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const session = await getServerSession(req, res, authOptions)
    if (!session) {
      return res.status(401).json({ error: 'Unauthorized' })
    }

    const validation = chatSchema.safeParse(req.body)
    if (!validation.success) {
      return res.status(400).json({
        error: 'Invalid chat data',
        details: validation.error.issues
      })
    }

    const { messages } = validation.data

    // 调用RAG服务的聊天接口
    const chatResult = await ragService.chat(messages)

    res.status(200).json({
      response: chatResult.response,
      context: chatResult.context,
      sources: chatResult.sources || [],
    })

  } catch (error) {
    console.error('Chat error:', error)
    res.status(500).json({
      error: 'Chat failed',
      details: error instanceof Error ? error.message : 'Unknown error'
    })
  }
}