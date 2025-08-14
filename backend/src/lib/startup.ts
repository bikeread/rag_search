import { rabbitMQService } from './rabbitmq'

let isInitialized = false

export async function initializeServices(): Promise<void> {
  if (isInitialized) {
    return
  }

  console.log('Initializing backend services...')

  try {
    // 初始化RabbitMQ服务
    await rabbitMQService.connect()
    
    isInitialized = true
    console.log('All backend services initialized successfully')
  } catch (error) {
    console.error('Service initialization failed:', error)
    throw error
  }
}

// 优雅关闭
export async function shutdownServices(): Promise<void> {
  console.log('Shutting down services...')
  
  try {
    await rabbitMQService.disconnect()
    console.log('Services shut down successfully')
  } catch (error) {
    console.error('Service shutdown error:', error)
  }
}

// 监听进程退出事件
if (typeof process !== 'undefined') {
  process.on('SIGTERM', shutdownServices)
  process.on('SIGINT', shutdownServices)
  process.on('beforeExit', shutdownServices)
}