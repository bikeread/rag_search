/**
 * 生产环境兼容的CORS配置
 */

// 允许的源域名配置
const getAllowedOrigins = (): string[] => {
  const origins = [
    // 开发环境
    'http://localhost:3000',
    'http://127.0.0.1:3000',
    
    // 从环境变量读取生产域名
    process.env.FRONTEND_URL,
    process.env.NEXT_PUBLIC_FRONTEND_URL,
    process.env.PRODUCTION_URL,
    
    // 可以添加多个生产域名
    'https://yourapp.com',
    'https://www.yourapp.com'
  ]
  
  // 过滤掉空值
  return origins.filter(Boolean) as string[]
}

/**
 * 获取CORS响应头
 */
export function getCorsHeaders(origin?: string | null): Record<string, string> {
  const allowedOrigins = getAllowedOrigins()
  
  // 检查origin是否在允许列表中
  const isAllowedOrigin = origin && allowedOrigins.includes(origin)
  
  return {
    'Access-Control-Allow-Origin': isAllowedOrigin ? origin : allowedOrigins[0],
    'Access-Control-Allow-Credentials': 'true',
    'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,PATCH,OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type,Authorization,X-Requested-With,X-CSRF-Token,Accept,Accept-Version,Content-Length,Content-MD5,Date,X-Api-Version'
  }
}

/**
 * 检查是否为允许的源
 */
export function isAllowedOrigin(origin?: string | null): boolean {
  if (!origin) return false
  const allowedOrigins = getAllowedOrigins()
  return allowedOrigins.includes(origin)
}

/**
 * 获取当前环境的前端URL
 */
export function getFrontendUrl(): string {
  return process.env.FRONTEND_URL || 
         process.env.NEXT_PUBLIC_FRONTEND_URL || 
         'http://localhost:3000'
}

/**
 * 通用的API CORS处理器
 * 用于包装API处理函数，自动处理CORS
 */
import { NextApiRequest, NextApiResponse } from 'next'
import { withJwtAuth, AuthenticatedRequest } from './jwtAuth'

type ApiHandler = (req: NextApiRequest, res: NextApiResponse) => Promise<void> | void
type AuthenticatedApiHandler = (req: AuthenticatedRequest, res: NextApiResponse) => Promise<void> | void

export function withCors(handler: ApiHandler) {
  return async (req: NextApiRequest, res: NextApiResponse) => {
    const origin = req.headers.origin
    
    // 添加CORS头
    const corsHeaders = getCorsHeaders(origin)
    Object.entries(corsHeaders).forEach(([key, value]) => {
      res.setHeader(key, value)
    })
    
    // 处理OPTIONS预检请求
    if (req.method === 'OPTIONS') {
      res.status(200).end()
      return
    }
    
    // 调用原始处理函数
    return handler(req, res)
  }
}

/**
 * 组合包装器：CORS + JWT认证
 * 最常用的API包装器组合
 */
export function withCorsAndAuth(handler: AuthenticatedApiHandler) {
  return withCors(withJwtAuth(handler))
}