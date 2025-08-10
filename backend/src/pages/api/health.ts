import { NextApiRequest, NextApiResponse } from 'next'
import { prisma } from '@/lib/prisma'
import { redis } from '@/lib/redis'
import { documentProcessor, vectorService, ragService } from '@/services/pythonServices'
import { getCorsHeaders } from '@/lib/cors'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const origin = req.headers.origin
  
  // 处理OPTIONS预检请求
  if (req.method === 'OPTIONS') {
    const corsHeaders = getCorsHeaders(origin)
    Object.entries(corsHeaders).forEach(([key, value]) => {
      res.setHeader(key, value)
    })
    res.status(200).end()
    return
  }

  // 为所有请求添加CORS头
  const corsHeaders = getCorsHeaders(origin)
  Object.entries(corsHeaders).forEach(([key, value]) => {
    res.setHeader(key, value)
  })

  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  const healthChecks = {
    database: false,
    redis: false,
    documentProcessor: false,
    vectorService: false,
    ragService: false,
  }

  let overallStatus = 'healthy'
  const errors: string[] = []

  // 检查数据库
  try {
    await prisma.$queryRaw`SELECT 1`
    healthChecks.database = true
  } catch (error) {
    errors.push(`Database: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }

  // 检查Redis
  try {
    await redis.ping()
    healthChecks.redis = true
  } catch (error) {
    errors.push(`Redis: ${error instanceof Error ? error.message : 'Unknown error'}`)
  }

  // 检查Python服务
  const pythonServices = [
    { name: 'documentProcessor', client: documentProcessor },
    { name: 'vectorService', client: vectorService },
    { name: 'ragService', client: ragService },
  ]

  await Promise.all(
    pythonServices.map(async ({ name, client }) => {
      try {
        await client.healthCheck()
        healthChecks[name as keyof typeof healthChecks] = true
      } catch (error) {
        errors.push(`${name}: ${error instanceof Error ? error.message : 'Unknown error'}`)
      }
    })
  )

  // 确定整体状态
  const healthyServices = Object.values(healthChecks).filter(Boolean).length
  const totalServices = Object.keys(healthChecks).length

  if (healthyServices === totalServices) {
    overallStatus = 'healthy'
  } else if (healthyServices >= totalServices * 0.7) {
    overallStatus = 'degraded'
  } else {
    overallStatus = 'unhealthy'
  }

  const statusCode = overallStatus === 'healthy' ? 200 : 
                     overallStatus === 'degraded' ? 207 : 503

  res.status(statusCode).json({
    status: overallStatus,
    timestamp: new Date().toISOString(),
    services: healthChecks,
    healthyServices,
    totalServices,
    errors: errors.length > 0 ? errors : undefined,
  })
}