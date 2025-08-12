import type { NextApiRequest, NextApiResponse } from 'next'
import jwt from 'jsonwebtoken'
import { prisma } from '@/lib/prisma'
import { getCorsHeaders } from '@/lib/cors'

// 确保应用服务已初始化
import '@/pages/api/_app-init'

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
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

  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const authorization = req.headers.authorization
    if (!authorization || !authorization.startsWith('Bearer ')) {
      return res.status(401).json({ error: 'No token provided' })
    }

    const token = authorization.substring(7)
    const JWT_SECRET = process.env.JWT_SECRET || 'fallback-secret-key'
    
    const decoded = jwt.verify(token, JWT_SECRET) as any
    
    const user = await prisma.user.findUnique({
      where: { id: decoded.userId },
      select: {
        id: true,
        email: true,
        name: true,
        role: true,
        createdAt: true,
        updatedAt: true
      }
    })

    if (!user) {
      return res.status(404).json({ error: 'User not found' })
    }

    res.status(200).json(user)
  } catch (error) {
    console.error('Get user error:', error)
    res.status(401).json({ error: 'Invalid token' })
  }
}