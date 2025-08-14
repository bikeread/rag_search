import { NextApiRequest, NextApiResponse } from 'next'
import jwt from 'jsonwebtoken'

const JWT_SECRET = process.env.JWT_SECRET || 'fallback-secret-key'
const JWT_REFRESH_SECRET = process.env.JWT_REFRESH_SECRET || 'refresh-secret-key'
const ACCESS_TOKEN_EXPIRE = '15m' // 15分钟
const REFRESH_TOKEN_EXPIRE = '7d' // 7天

export interface AuthenticatedRequest extends NextApiRequest {
  user: {
    id: string
    email: string
    name?: string
    role?: string
  }
}

export interface JwtPayload {
  userId: string
  email: string
  name?: string
  role?: string
}

/**
 * 生成访问令牌
 */
export function generateAccessToken(payload: JwtPayload): string {
  return jwt.sign(payload, JWT_SECRET, { expiresIn: ACCESS_TOKEN_EXPIRE })
}

/**
 * 生成刷新令牌
 */
export function generateRefreshToken(payload: JwtPayload): string {
  return jwt.sign(payload, JWT_REFRESH_SECRET, { expiresIn: REFRESH_TOKEN_EXPIRE })
}

/**
 * 生成令牌对
 */
export function generateTokenPair(user: { id: string; email: string; name?: string; role?: string }) {
  const payload: JwtPayload = {
    userId: user.id,
    email: user.email,
    name: user.name,
    role: user.role
  }

  return {
    accessToken: generateAccessToken(payload),
    refreshToken: generateRefreshToken(payload),
    expiresIn: 900 // 15分钟（秒）
  }
}

/**
 * 验证刷新令牌
 */
export function verifyRefreshToken(token: string): JwtPayload | null {
  try {
    return jwt.verify(token, JWT_REFRESH_SECRET) as JwtPayload
  } catch {
    return null
  }
}

/**
 * JWT认证中间件
 * 验证Authorization header中的Bearer token
 * 支持自动刷新即将过期的令牌
 */
export function verifyJwtToken(req: NextApiRequest): { success: boolean; user?: any; error?: string; newToken?: string } {
  try {
    const authorization = req.headers.authorization
    
    if (!authorization || !authorization.startsWith('Bearer ')) {
      return { success: false, error: 'No token provided' }
    }

    const token = authorization.substring(7)
    
    const decoded = jwt.verify(token, JWT_SECRET) as JwtPayload
    
    // 检查令牌是否即将过期（少于5分钟）
    const tokenData = jwt.decode(token) as any
    const now = Math.floor(Date.now() / 1000)
    const timeLeft = tokenData.exp - now
    
    let newToken: string | undefined
    if (timeLeft < 300) { // 少于5分钟，生成新令牌
      newToken = generateAccessToken(decoded)
    }
    
    return { 
      success: true, 
      user: {
        id: decoded.userId,
        email: decoded.email,
        name: decoded.name,
        role: decoded.role
      },
      newToken
    }
  } catch (error) {
    return { 
      success: false, 
      error: error instanceof Error ? error.message : 'Invalid token' 
    }
  }
}

/**
 * 高阶函数：为API handler添加JWT认证
 */
export function withJwtAuth<T = any>(
  handler: (req: AuthenticatedRequest, res: NextApiResponse<T>) => Promise<void> | void
) {
  return async (req: NextApiRequest, res: NextApiResponse<T>) => {
    const authResult = verifyJwtToken(req)
    
    if (!authResult.success) {
      return res.status(401).json({ error: authResult.error } as T)
    }

    // 如果有新令牌，设置响应头
    if (authResult.newToken) {
      res.setHeader('X-New-Access-Token', authResult.newToken)
    }

    // 添加用户信息到请求对象
    ;(req as AuthenticatedRequest).user = authResult.user!
    
    return handler(req as AuthenticatedRequest, res)
  }
}