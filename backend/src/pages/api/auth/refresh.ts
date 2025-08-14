import { NextApiRequest, NextApiResponse } from 'next'
import { verifyRefreshToken, generateTokenPair } from '../../../lib/jwtAuth'
import { prisma } from '../../../lib/prisma'
import { withCors } from '../../../lib/cors'

async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const { refreshToken } = req.body

    if (!refreshToken) {
      return res.status(400).json({ error: 'Refresh token is required' })
    }

    // 验证刷新令牌
    const payload = verifyRefreshToken(refreshToken)
    
    if (!payload) {
      return res.status(401).json({ error: 'Invalid refresh token' })
    }

    // 检查用户是否仍然存在且有效
    const user = await prisma.user.findUnique({
      where: { id: payload.userId },
      select: { 
        id: true, 
        email: true, 
        name: true,
        role: true,
        isActive: true 
      }
    })

    if (!user || !user.isActive) {
      return res.status(401).json({ error: 'User not found or inactive' })
    }

    // 生成新的令牌对
    const tokens = generateTokenPair({
      id: user.id,
      email: user.email,
      name: user.name || undefined,
      role: user.role
    })

    return res.status(200).json({
      success: true,
      ...tokens
    })
  } catch (error) {
    console.error('Token refresh error:', error)
    return res.status(500).json({ error: 'Failed to refresh token' })
  }
}

export default withCors(handler)