import { NextApiRequest, NextApiResponse } from 'next'
import jwt from 'jsonwebtoken'

export interface AuthenticatedRequest extends NextApiRequest {
  user: {
    id: string
    email: string
    name?: string
  }
}

/**
 * JWT认证中间件
 * 验证Authorization header中的Bearer token
 */
export function verifyJwtToken(req: NextApiRequest): { success: boolean; user?: any; error?: string } {
  try {
    const authorization = req.headers.authorization
    
    if (!authorization || !authorization.startsWith('Bearer ')) {
      return { success: false, error: 'No token provided' }
    }

    const token = authorization.substring(7)
    const JWT_SECRET = process.env.JWT_SECRET || 'fallback-secret-key'
    
    const decoded = jwt.verify(token, JWT_SECRET) as any
    
    return { 
      success: true, 
      user: {
        id: decoded.userId,
        email: decoded.email,
        name: decoded.name
      }
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

    // 添加用户信息到请求对象
    ;(req as AuthenticatedRequest).user = authResult.user!
    
    return handler(req as AuthenticatedRequest, res)
  }
}