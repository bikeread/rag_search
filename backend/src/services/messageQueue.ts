import { RabbitMQService } from '@/lib/rabbitmq'

export const messageQueue = new RabbitMQService()

// 在应用启动时连接
messageQueue.connect().catch(console.error)

// 优雅关闭
process.on('SIGINT', async () => {
  await messageQueue.disconnect()
  process.exit(0)
})