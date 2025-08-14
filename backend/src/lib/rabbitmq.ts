import amqp from 'amqplib'
import { prisma } from './prisma'

interface ProcessedMessage {
  document_id: string
  status: 'completed' | 'failed'
  chunks?: Array<{
    content: string
    chunk_index: number
    vector_id: string
    metadata?: any
  }>
  error_message?: string
  processing_time?: number
}

export class RabbitMQService {
  private connection: amqp.Connection | null = null
  private channel: amqp.Channel | null = null
  private readonly url: string

  constructor() {
    this.url = process.env.RABBITMQ_URL || 'amqp://guest:guest@localhost:5672'
    console.log('RabbitMQ service initialized')
  }

  async connect(): Promise<void> {
    try {
      this.connection = await amqp.connect(this.url)
      this.channel = await this.connection.createChannel()
      
      // 确保交换机和队列存在
      await this.channel.assertExchange('rag_exchange', 'topic', { durable: true })
      await this.channel.assertQueue('document_processed', { durable: true })
      
      // 绑定队列到交换机的路由键
      await this.channel.bindQueue('document_processed', 'rag_exchange', 'rag.document_processed')
      
      console.log('RabbitMQ connected successfully')
      
      // 开始消费消息
      await this.startConsuming()
    } catch (error) {
      console.error('RabbitMQ connection failed:', error)
      throw error
    }
  }

  async disconnect(): Promise<void> {
    try {
      if (this.channel) {
        await this.channel.close()
        this.channel = null
      }
      if (this.connection) {
        await this.connection.close()
        this.connection = null
      }
      console.log('RabbitMQ disconnected')
    } catch (error) {
      console.error('RabbitMQ disconnect error:', error)
    }
  }

  private async startConsuming(): Promise<void> {
    if (!this.channel) {
      throw new Error('Channel not initialized')
    }

    await this.channel.consume('document_processed', async (msg) => {
      if (msg) {
        try {
          const message: ProcessedMessage = JSON.parse(msg.content.toString())
          await this.handleDocumentProcessed(message)
          this.channel!.ack(msg)
          console.log(`Processed message for document: ${message.document_id}`)
        } catch (error) {
          console.error('Message processing error:', error)
          this.channel!.nack(msg, false, false) // 发送到死信队列
        }
      }
    })

    console.log('Started consuming document_processed messages')
  }

  private async handleDocumentProcessed(message: ProcessedMessage): Promise<void> {
    const { document_id, status, chunks, error_message } = message
    console.log(`[RabbitMQ Debug] Processing message for document_id: ${document_id}`)

    if (status === 'completed' && chunks) {
      // 成功处理 - 更新文档状态并写入chunks
      await prisma.$transaction(async (tx) => {
        // 更新文档状态
        await tx.document.update({
          where: { id: document_id },
          data: {
            status: 'COMPLETED',
            processingCompletedAt: new Date(),
            errorMessage: null
          }
        })

        // 写入chunks数据
        if (chunks && chunks.length > 0) {
          await tx.documentChunk.createMany({
            data: chunks.map((chunk, index) => ({
              documentId: document_id,
              content: chunk.content,
              chunkIndex: chunk.chunk_index || index,
              vectorId: chunk.vector_id || `no-vector-${document_id}-${index}`,
              metadata: chunk.metadata || {}
            }))
          })
        }
      })

      console.log(`Document ${document_id} processed successfully with ${chunks.length} chunks`)
    } else {
      // 处理失败
      await prisma.document.update({
        where: { id: document_id },
        data: {
          status: 'FAILED',
          errorMessage: error_message || 'Processing failed',
          processingCompletedAt: new Date()
        }
      })

      console.log(`Document ${document_id} processing failed: ${error_message}`)
    }
  }
}

// 单例实例
export const rabbitMQService = new RabbitMQService()