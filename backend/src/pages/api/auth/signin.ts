import { NextApiRequest, NextApiResponse } from 'next'
import bcrypt from 'bcryptjs'
import { prisma } from '@/lib/prisma'
import { z } from 'zod'
import { getCorsHeaders } from '@/lib/cors'
import { generateTokenPair } from '@/lib/jwtAuth'
import { strictLimiter } from '../../../middleware/security'
import { sanitizeInput } from '@/lib/validation'

const signinSchema = z.object({
  email: z.string().email(),
  password: z.string().min(1),
})

async function signinHandler(req: NextApiRequest, res: NextApiResponse) {
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
    const validation = signinSchema.safeParse(req.body)
    
    if (!validation.success) {
      return res.status(400).json({
        error: 'Invalid data',
        details: validation.error.issues
      })
    }

    // 清理输入防止SQL注入
    const email = sanitizeInput(validation.data.email)
    const { password } = validation.data

    // Find user by email
    const user = await prisma.user.findUnique({
      where: { email }
    })

    if (!user || !user.password) {
      return res.status(401).json({ error: 'Invalid credentials' })
    }

    // Verify password
    const isPasswordValid = await bcrypt.compare(password, user.password)

    if (!isPasswordValid) {
      return res.status(401).json({ error: 'Invalid credentials' })
    }

    // 生成访问令牌和刷新令牌
    const tokens = generateTokenPair({
      id: user.id,
      email: user.email,
      name: user.name || undefined,
      role: user.role
    })

    res.status(200).json({
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        role: user.role,
      },
      token: tokens.accessToken,
      refreshToken: tokens.refreshToken,
      expiresIn: tokens.expiresIn
    })

  } catch (error) {
    console.error('Signin error:', error)
    res.status(500).json({ error: 'Signin failed' })
  }
}

// 应用速率限制中间件
export default function handler(req: NextApiRequest, res: NextApiResponse) {
  // 应用严格限流（15分钟内最多5次尝试）
  return new Promise((resolve) => {
    strictLimiter(req as any, res as any, () => {
      resolve(signinHandler(req, res))
    })
  })
}