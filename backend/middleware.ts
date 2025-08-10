import { NextRequest, NextResponse } from 'next/server'

/**
 * 生产环境兼容的CORS配置
 */
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
function getCorsHeaders(origin?: string | null): Record<string, string> {
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
 * Next.js Middleware - 全局处理CORS
 * 这个中间件会在所有API请求之前执行
 */
export function middleware(request: NextRequest) {
  const origin = request.headers.get('origin')
  
  // 添加调试日志
  console.log(`[Middleware] ${request.method} ${request.url} from ${origin}`)
  
  // 处理OPTIONS预检请求
  if (request.method === 'OPTIONS') {
    console.log('[Middleware] Handling OPTIONS preflight request')
    return new NextResponse(null, {
      status: 200,
      headers: getCorsHeaders(origin)
    })
  }

  // 对于其他请求，继续正常处理并添加CORS头
  const response = NextResponse.next()
  
  // 为所有API响应添加CORS头
  const corsHeaders = getCorsHeaders(origin)
  Object.entries(corsHeaders).forEach(([key, value]) => {
    response.headers.set(key, value)
  })
  
  console.log('[Middleware] Added CORS headers to response')
  return response
}

// 配置中间件匹配路径
export const config = {
  matcher: [
    '/api/(.*)',  // 使用正则表达式匹配所有API路由
  ]
}