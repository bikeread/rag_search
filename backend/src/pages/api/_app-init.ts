import { NextApiRequest, NextApiResponse } from 'next'
import { initializeServices } from '@/lib/startup'

let initialized = false

// 应用启动时自动初始化服务
const initialize = async () => {
  if (!initialized) {
    console.log('Starting application service initialization...')
    try {
      await initializeServices()
      initialized = true
      console.log('Application services initialized successfully')
    } catch (error) {
      console.error('Failed to initialize application services:', error)
      // 在开发环境中，服务初始化失败不应该阻止应用启动
      if (process.env.NODE_ENV !== 'development') {
        throw error
      }
    }
  }
}

// 立即执行初始化（当模块被加载时）
initialize().catch(console.error)

// 提供健康检查接口
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  return res.status(200).json({
    initialized,
    timestamp: new Date().toISOString(),
    message: initialized ? 'Services initialized' : 'Services not initialized'
  })
}