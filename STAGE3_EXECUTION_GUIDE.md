# 阶段3执行指导文档：Next.js后端开发

## 📋 总览信息

**目标**: 构建完整的业务逻辑和API编排层，连接前端与Python AI服务
**预计时间**: 12-15天 (2-2.5周)
**执行策略**: 自底向上开发 (基础设施→API→业务逻辑)

## 🎯 核心目标

- ✅ **Next.js API Routes** - 完整的RESTful API接口
- ✅ **数据库集成** - PostgreSQL + Prisma ORM
- ✅ **Python服务编排** - 与3个AI微服务的通信
- ✅ **认证授权系统** - NextAuth.js用户管理
- ✅ **缓存系统** - Redis缓存优化
- ✅ **消息队列处理** - RabbitMQ异步消息处理
- ✅ **错误处理和监控** - 完善的错误处理机制

---

## 🏗️ 子阶段拆分执行计划

### 子阶段3.1：项目基础设施搭建 (3-4天) 🔧

**状态**: 🔴 待开始  
**优先级**: 🔥 最高 (阻塞其他任务)  
**依赖**: 阶段2完成 (Python微服务就绪)

#### 📁 目录结构创建

```bash
backend/                           # Next.js后端项目
├── src/
│   ├── pages/
│   │   ├── api/                   # API路由
│   │   │   ├── auth/              # 认证相关API
│   │   │   │   ├── [...nextauth].ts
│   │   │   │   └── register.ts
│   │   │   ├── documents/         # 文档管理API
│   │   │   │   ├── upload.ts
│   │   │   │   ├── status/[id].ts
│   │   │   │   ├── list.ts
│   │   │   │   └── delete/[id].ts
│   │   │   ├── query/             # RAG查询API
│   │   │   │   ├── index.ts
│   │   │   │   ├── history.ts
│   │   │   │   └── chat.ts
│   │   │   ├── admin/             # 管理员API
│   │   │   │   ├── users.ts
│   │   │   │   ├── documents.ts
│   │   │   │   └── system.ts
│   │   │   └── health.ts          # 健康检查
│   │   └── _app.ts
│   ├── lib/                       # 核心库
│   │   ├── auth.ts                # NextAuth配置
│   │   ├── prisma.ts              # Prisma客户端
│   │   ├── redis.ts               # Redis客户端
│   │   ├── rabbitmq.ts            # RabbitMQ客户端
│   │   └── validation.ts          # 数据验证
│   ├── services/                  # 业务服务
│   │   ├── pythonServices.ts      # Python服务客户端
│   │   ├── documentService.ts     # 文档业务逻辑
│   │   ├── queryService.ts        # 查询业务逻辑
│   │   ├── userService.ts         # 用户业务逻辑
│   │   └── messageQueue.ts        # 消息队列服务
│   ├── middleware/                # 中间件
│   │   ├── auth.ts                # 认证中间件
│   │   ├── cors.ts                # CORS中间件
│   │   ├── rateLimit.ts           # 限流中间件
│   │   └── monitoring.ts          # 监控中间件
│   ├── types/                     # TypeScript类型定义
│   │   ├── api.ts                 # API类型
│   │   ├── database.ts            # 数据库类型
│   │   └── python-services.ts     # Python服务类型
│   └── utils/                     # 工具函数
│       ├── errors.ts              # 错误处理
│       ├── logger.ts              # 日志工具
│       └── helpers.ts             # 通用工具
├── prisma/                        # 数据库Schema
│   ├── schema.prisma
│   ├── migrations/
│   └── seed.ts
├── package.json
├── next.config.js
├── tsconfig.json
├── .env.example
├── .env.local
└── Dockerfile
```

#### 🎯 具体任务清单

**Task 3.1.1: Next.js项目初始化**

```bash
# 创建Next.js项目
npx create-next-app@latest backend --typescript --tailwind --eslint --app --src-dir --import-alias "@/*"

# 安装核心依赖
npm install prisma @prisma/client next-auth @next-auth/prisma-adapter
npm install ioredis amqplib bcryptjs jsonwebtoken
npm install zod formidable multer sharp
npm install @types/bcryptjs @types/jsonwebtoken @types/formidable @types/multer

# 安装开发依赖
npm install -D @types/node typescript tsx nodemon
```

**Task 3.1.2: 数据库Schema设计**

```prisma
// prisma/schema.prisma
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model User {
  id        String   @id @default(cuid())
  email     String   @unique
  name      String?
  password  String?  // 用于本地认证
  role      UserRole @default(USER)
  avatar    String?
  isActive  Boolean  @default(true)
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt

  // 关联关系
  documents Document[]
  queries   Query[]
  sessions  Session[]
  accounts  Account[]

  @@map("users")
}

model Account {
  id                String  @id @default(cuid())
  userId            String
  type              String
  provider          String
  providerAccountId String
  refresh_token     String? @db.Text
  access_token      String? @db.Text
  expires_at        Int?
  token_type        String?
  scope             String?
  id_token          String? @db.Text
  session_state     String?

  user User @relation(fields: [userId], references: [id], onDelete: Cascade)

  @@unique([provider, providerAccountId])
  @@map("accounts")
}

model Session {
  id           String   @id @default(cuid())
  sessionToken String   @unique
  userId       String
  expires      DateTime
  user         User     @relation(fields: [userId], references: [id], onDelete: Cascade)

  @@map("sessions")
}

model Document {
  id            String          @id @default(cuid())
  filename      String
  originalName  String
  mimeType      String
  size          Int
  path          String?         // 文件存储路径
  status        DocumentStatus  @default(PENDING)
  uploadedBy    String
  processingStartedAt DateTime?
  processingCompletedAt DateTime?
  errorMessage  String?
  metadata      Json?           // 额外元数据
  createdAt     DateTime        @default(now())
  updatedAt     DateTime        @updatedAt

  // 关联关系
  user          User            @relation(fields: [uploadedBy], references: [id])
  chunks        DocumentChunk[]

  @@map("documents")
}

model DocumentChunk {
  id          String   @id @default(cuid())
  documentId  String
  content     String   @db.Text
  chunkIndex  Int
  vectorId    String   // Milvus向量ID
  metadata    Json?
  createdAt   DateTime @default(now())

  // 关联关系
  document    Document @relation(fields: [documentId], references: [id], onDelete: Cascade)

  @@unique([documentId, chunkIndex])
  @@map("document_chunks")
}

model Query {
  id          String    @id @default(cuid())
  text        String    @db.Text
  response    String?   @db.Text
  userId      String
  responseTime Int?     // 响应时间(ms)
  sources     Json?     // 检索到的源文档
  status      QueryStatus @default(PENDING)
  metadata    Json?     // 查询元数据
  createdAt   DateTime  @default(now())

  // 关联关系
  user        User      @relation(fields: [userId], references: [id])

  @@map("queries")
}

model SystemConfig {
  id        String   @id @default(cuid())
  key       String   @unique
  value     Json
  description String?
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt

  @@map("system_config")
}

// 枚举类型
enum UserRole {
  USER
  ADMIN
  SUPER_ADMIN
}

enum DocumentStatus {
  PENDING      // 等待处理
  PROCESSING   // 处理中
  COMPLETED    // 处理完成
  FAILED       // 处理失败
  DELETED      // 已删除
}

enum QueryStatus {
  PENDING      // 等待处理
  PROCESSING   // 处理中
  COMPLETED    // 完成
  FAILED       // 失败
}
```

**Task 3.1.3: 基础配置文件**

```typescript
// next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
  },
  api: {
    bodyParser: {
      sizeLimit: '10mb',
    },
  },
  env: {
    DOCUMENT_PROCESSOR_URL: process.env.DOCUMENT_PROCESSOR_URL,
    VECTOR_SERVICE_URL: process.env.VECTOR_SERVICE_URL,
    RAG_SERVICE_URL: process.env.RAG_SERVICE_URL,
  },
}

module.exports = nextConfig

// .env.example
# Database
DATABASE_URL="postgresql://postgres:password@localhost:5432/rag_db"
REDIS_URL="redis://localhost:6379"
RABBITMQ_URL="amqp://guest:guest@localhost:5672"

# NextAuth
NEXTAUTH_URL="http://localhost:3001"
NEXTAUTH_SECRET="your-nextauth-secret-here"

# Python Services
DOCUMENT_PROCESSOR_URL="http://localhost:8001"
VECTOR_SERVICE_URL="http://localhost:8002"
RAG_SERVICE_URL="http://localhost:8003"

# Upload
MAX_FILE_SIZE="10485760"  # 10MB
UPLOAD_DIR="./uploads"

# Admin
ADMIN_EMAIL="admin@example.com"
ADMIN_PASSWORD="admin123"
```

#### ✅ 验收标准3.1

- [ ] Next.js项目成功创建并可启动
- [ ] 数据库连接正常，Prisma生成成功
- [ ] 基础目录结构创建完成
- [ ] 环境变量配置正确
- [ ] TypeScript配置无错误

---

### 子阶段3.2：核心基础设施服务 (3-4天) 🔌

**状态**: 🔴 待开始  
**优先级**: 🔥 高  
**依赖**: 子阶段3.1完成

#### 🎯 具体任务清单

**Task 3.2.1: 数据库客户端配置**

```typescript
// src/lib/prisma.ts
import { PrismaClient } from '@prisma/client'

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined
}

export const prisma = globalForPrisma.prisma ?? new PrismaClient()

if (process.env.NODE_ENV !== 'production') globalForPrisma.prisma = prisma

// src/lib/redis.ts
import Redis from 'ioredis'

const getRedisUrl = () => {
  if (process.env.REDIS_URL) {
    return process.env.REDIS_URL
  }
  throw new Error('REDIS_URL is not defined')
}

export const redis = new Redis(getRedisUrl(), {
  retryDelayOnFailover: 100,
  enableReadyCheck: false,
  maxRetriesPerRequest: null,
})

// 缓存工具函数
export class CacheService {
  private static DEFAULT_TTL = 3600 // 1小时

  static async get<T>(key: string): Promise<T | null> {
    try {
      const value = await redis.get(key)
      return value ? JSON.parse(value) : null
    } catch (error) {
      console.error('Cache get error:', error)
      return null
    }
  }

  static async set(key: string, value: any, ttl: number = CacheService.DEFAULT_TTL): Promise<void> {
    try {
      await redis.setex(key, ttl, JSON.stringify(value))
    } catch (error) {
      console.error('Cache set error:', error)
    }
  }

  static async del(key: string): Promise<void> {
    try {
      await redis.del(key)
    } catch (error) {
      console.error('Cache delete error:', error)
    }
  }

  static async exists(key: string): Promise<boolean> {
    try {
      const result = await redis.exists(key)
      return result === 1
    } catch (error) {
      console.error('Cache exists error:', error)
      return false
    }
  }
}
```

**Task 3.2.2: NextAuth认证系统**

```typescript
// src/lib/auth.ts
import { NextAuthOptions } from "next-auth"
import { PrismaAdapter } from "@next-auth/prisma-adapter"
import CredentialsProvider from "next-auth/providers/credentials"
import bcrypt from "bcryptjs"
import { prisma } from "./prisma"

export const authOptions: NextAuthOptions = {
  adapter: PrismaAdapter(prisma),
  providers: [
    CredentialsProvider({
      name: "credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" }
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          return null
        }

        const user = await prisma.user.findUnique({
          where: {
            email: credentials.email
          }
        })

        if (!user || !user.password) {
          return null
        }

        const isPasswordValid = await bcrypt.compare(
          credentials.password,
          user.password
        )

        if (!isPasswordValid) {
          return null
        }

        return {
          id: user.id,
          email: user.email,
          name: user.name,
          role: user.role,
        }
      }
    })
  ],
  session: {
    strategy: "jwt"
  },
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.role = user.role
      }
      return token
    },
    async session({ session, token }) {
      if (token) {
        session.user.id = token.sub!
        session.user.role = token.role as string
      }
      return session
    }
  },
  pages: {
    signIn: "/auth/signin",
    signUp: "/auth/signup",
  }
}

// src/pages/api/auth/[...nextauth].ts
import NextAuth from "next-auth"
import { authOptions } from "@/lib/auth"

export default NextAuth(authOptions)

// src/pages/api/auth/register.ts
import { NextApiRequest, NextApiResponse } from 'next'
import bcrypt from 'bcryptjs'
import { prisma } from '@/lib/prisma'
import { registerSchema } from '@/lib/validation'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const { email, password, name } = registerSchema.parse(req.body)

    // 检查用户是否已存在
    const existingUser = await prisma.user.findUnique({
      where: { email }
    })

    if (existingUser) {
      return res.status(400).json({ error: 'User already exists' })
    }

    // 加密密码
    const hashedPassword = await bcrypt.hash(password, 12)

    // 创建用户
    const user = await prisma.user.create({
      data: {
        email,
        password: hashedPassword,
        name,
      }
    })

    res.status(201).json({
      message: 'User created successfully',
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
      }
    })

  } catch (error) {
    console.error('Registration error:', error)
    res.status(500).json({ error: 'Registration failed' })
  }
}
```

**Task 3.2.3: Python服务客户端**

```typescript
// src/services/pythonServices.ts
import axios, { AxiosInstance, AxiosRequestConfig } from 'axios'

// 服务配置
const SERVICE_URLS = {
  DOCUMENT_PROCESSOR: process.env.DOCUMENT_PROCESSOR_URL || 'http://localhost:8001',
  VECTOR_SERVICE: process.env.VECTOR_SERVICE_URL || 'http://localhost:8002',
  RAG_SERVICE: process.env.RAG_SERVICE_URL || 'http://localhost:8003',
}

// 通用HTTP客户端类
class PythonServiceClient {
  private client: AxiosInstance

  constructor(baseURL: string, timeout: number = 30000) {
    this.client = axios.create({
      baseURL,
      timeout,
      headers: {
        'Content-Type': 'application/json',
      },
    })

    // 请求拦截器
    this.client.interceptors.request.use(
      (config) => {
        console.log(`[${new Date().toISOString()}] ${config.method?.toUpperCase()} ${config.url}`)
        return config
      },
      (error) => Promise.reject(error)
    )

    // 响应拦截器
    this.client.interceptors.response.use(
      (response) => {
        console.log(`[${new Date().toISOString()}] Response: ${response.status}`)
        return response
      },
      (error) => {
        console.error(`[${new Date().toISOString()}] Error: ${error.message}`)
        return Promise.reject(error)
      }
    )
  }

  async get(url: string, config?: AxiosRequestConfig) {
    const response = await this.client.get(url, config)
    return response.data
  }

  async post(url: string, data?: any, config?: AxiosRequestConfig) {
    const response = await this.client.post(url, data, config)
    return response.data
  }

  async put(url: string, data?: any, config?: AxiosRequestConfig) {
    const response = await this.client.put(url, data, config)
    return response.data
  }

  async delete(url: string, config?: AxiosRequestConfig) {
    const response = await this.client.delete(url, config)
    return response.data
  }
}

// 文档处理服务客户端
export class DocumentProcessorClient extends PythonServiceClient {
  constructor() {
    super(SERVICE_URLS.DOCUMENT_PROCESSOR)
  }

  async uploadDocument(file: Buffer, filename: string, documentId: string) {
    const formData = new FormData()
    formData.append('file', new Blob([file]), filename)
    formData.append('document_id', documentId)

    return this.post('/process-document', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
  }

  async getProcessingStatus(documentId: string) {
    return this.get(`/processing-status/${documentId}`)
  }

  async reprocessDocument(documentId: string) {
    return this.post(`/reprocess-document/${documentId}`)
  }

  async healthCheck() {
    return this.get('/health')
  }
}

// 向量化服务客户端
export class VectorServiceClient extends PythonServiceClient {
  constructor() {
    super(SERVICE_URLS.VECTOR_SERVICE)
  }

  async vectorizeTexts(texts: string[], storeVectors: boolean = true) {
    return this.post('/vectorize', {
      texts,
      store_vectors: storeVectors
    })
  }

  async searchVectors(queryText: string, topK: number = 5) {
    return this.post('/search', {
      query_text: queryText,
      top_k: topK
    })
  }

  async storeVectors(vectors: number[][], texts: string[], metadata?: any[]) {
    return this.post('/store-vectors', {
      vectors,
      texts,
      metadata
    })
  }

  async healthCheck() {
    return this.get('/health')
  }
}

// RAG服务客户端
export class RAGServiceClient extends PythonServiceClient {
  constructor() {
    super(SERVICE_URLS.RAG_SERVICE)
  }

  async query(query: string, topK: number = 5, includeSources: boolean = true) {
    return this.post('/query', {
      query,
      top_k: topK,
      include_sources: includeSources
    })
  }

  async chat(messages: Array<{role: string, content: string}>) {
    return this.post('/chat', {
      messages
    })
  }

  async documentQA(documentId: string, question: string) {
    return this.get(`/documents/${documentId}/qa`, {
      params: { question }
    })
  }

  async healthCheck() {
    return this.get('/health')
  }
}

// 导出客户端实例
export const documentProcessor = new DocumentProcessorClient()
export const vectorService = new VectorServiceClient()
export const ragService = new RAGServiceClient()
```

**Task 3.2.4: RabbitMQ消息队列服务**

```typescript
// src/lib/rabbitmq.ts
import amqp, { Connection, Channel } from 'amqplib'

export class RabbitMQService {
  private connection: Connection | null = null
  private channel: Channel | null = null
  private url: string

  constructor() {
    this.url = process.env.RABBITMQ_URL || 'amqp://localhost:5672'
  }

  async connect(): Promise<void> {
    try {
      this.connection = await amqp.connect(this.url)
      this.channel = await this.connection.createChannel()
      
      // 声明队列
      await this.setupQueues()
      
      console.log('Connected to RabbitMQ')
    } catch (error) {
      console.error('RabbitMQ connection failed:', error)
      throw error
    }
  }

  private async setupQueues(): Promise<void> {
    if (!this.channel) throw new Error('Channel not initialized')

    // 文档处理队列
    await this.channel.assertQueue('document_processed', { durable: true })
    await this.channel.assertQueue('vectorization_completed', { durable: true })
    
    // 消费消息
    await this.startConsumers()
  }

  private async startConsumers(): Promise<void> {
    if (!this.channel) return

    // 监听文档处理完成消息
    await this.channel.consume('document_processed', async (msg) => {
      if (msg) {
        try {
          const data = JSON.parse(msg.content.toString())
          await this.handleDocumentProcessed(data)
          this.channel!.ack(msg)
        } catch (error) {
          console.error('Error processing document message:', error)
          this.channel!.nack(msg, false, false)
        }
      }
    })

    // 监听向量化完成消息
    await this.channel.consume('vectorization_completed', async (msg) => {
      if (msg) {
        try {
          const data = JSON.parse(msg.content.toString())
          await this.handleVectorizationCompleted(data)
          this.channel!.ack(msg)
        } catch (error) {
          console.error('Error processing vectorization message:', error)
          this.channel!.nack(msg, false, false)
        }
      }
    })
  }

  private async handleDocumentProcessed(data: any): Promise<void> {
    const { document_id, status, chunks, error_message } = data

    if (status === 'completed') {
      // 更新文档状态为已完成
      await prisma.document.update({
        where: { id: document_id },
        data: {
          status: 'COMPLETED',
          processingCompletedAt: new Date(),
        }
      })

      // 保存文档块
      if (chunks && chunks.length > 0) {
        await prisma.documentChunk.createMany({
          data: chunks.map((chunk: any, index: number) => ({
            documentId: document_id,
            content: chunk.content,
            chunkIndex: index,
            vectorId: chunk.vector_id || '',
            metadata: chunk.metadata || {},
          }))
        })
      }
    } else if (status === 'failed') {
      await prisma.document.update({
        where: { id: document_id },
        data: {
          status: 'FAILED',
          errorMessage: error_message,
        }
      })
    }
  }

  private async handleVectorizationCompleted(data: any): Promise<void> {
    const { document_id, vector_count, status } = data

    console.log(`Vectorization completed for document ${document_id}: ${vector_count} vectors`)
    
    // 可以在这里添加额外的处理逻辑
    // 比如发送通知、更新统计信息等
  }

  async disconnect(): Promise<void> {
    if (this.channel) {
      await this.channel.close()
    }
    if (this.connection) {
      await this.connection.close()
    }
  }
}

// src/services/messageQueue.ts
import { RabbitMQService } from '@/lib/rabbitmq'

export const messageQueue = new RabbitMQService()

// 在应用启动时连接
messageQueue.connect().catch(console.error)

// 优雅关闭
process.on('SIGINT', async () => {
  await messageQueue.disconnect()
  process.exit(0)
})
```

#### ✅ 验收标准3.2

- [ ] 数据库连接和操作正常
- [ ] Redis缓存服务正常工作
- [ ] NextAuth认证系统配置完成
- [ ] Python服务客户端通信正常
- [ ] RabbitMQ消息队列处理正常
- [ ] 所有基础服务健康检查通过

---

### 子阶段3.3：文档管理API开发 (3-4天) 📄

**状态**: 🔴 待开始  
**优先级**: ⚡ 中高  
**依赖**: 子阶段3.2完成

#### 🎯 具体任务清单

**Task 3.3.1: 文档上传API**

```typescript
// src/pages/api/documents/upload.ts
import { NextApiRequest, NextApiResponse } from 'next'
import { getServerSession } from 'next-auth/next'
import formidable from 'formidable'
import fs from 'fs/promises'
import path from 'path'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/prisma'
import { documentProcessor } from '@/services/pythonServices'
import { uploadSchema } from '@/lib/validation'

export const config = {
  api: {
    bodyParser: false,
  },
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    // 验证用户认证
    const session = await getServerSession(req, res, authOptions)
    if (!session) {
      return res.status(401).json({ error: 'Unauthorized' })
    }

    // 解析上传文件
    const form = formidable({
      multiples: false,
      maxFileSize: parseInt(process.env.MAX_FILE_SIZE || '10485760'), // 10MB
      filter: ({ mimetype }) => {
        // 允许的文件类型
        const allowedTypes = [
          'application/pdf',
          'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
          'application/msword',
          'text/plain',
          'text/markdown',
        ]
        return allowedTypes.includes(mimetype || '')
      }
    })

    const [fields, files] = await form.parse(req)
    const file = Array.isArray(files.file) ? files.file[0] : files.file

    if (!file) {
      return res.status(400).json({ error: 'No file uploaded' })
    }

    // 验证文件
    const validation = uploadSchema.safeParse({
      filename: file.originalFilename,
      size: file.size,
      mimeType: file.mimetype,
    })

    if (!validation.success) {
      return res.status(400).json({ 
        error: 'Invalid file',
        details: validation.error.errors
      })
    }

    // 创建文档记录
    const document = await prisma.document.create({
      data: {
        filename: file.originalFilename || 'unknown',
        originalName: file.originalFilename || 'unknown',
        mimeType: file.mimetype || 'application/octet-stream',
        size: file.size,
        status: 'PENDING',
        uploadedBy: session.user.id,
        processingStartedAt: new Date(),
      }
    })

    // 读取文件内容
    const fileBuffer = await fs.readFile(file.filepath)

    // 发送到文档处理服务
    try {
      const processResult = await documentProcessor.uploadDocument(
        fileBuffer,
        file.originalFilename || 'unknown',
        document.id
      )

      // 更新状态为处理中
      await prisma.document.update({
        where: { id: document.id },
        data: { status: 'PROCESSING' }
      })

      res.status(200).json({
        documentId: document.id,
        status: 'processing',
        message: 'Document uploaded and processing started',
        processingInfo: processResult
      })

    } catch (processingError) {
      // 处理失败，更新状态
      await prisma.document.update({
        where: { id: document.id },
        data: {
          status: 'FAILED',
          errorMessage: processingError instanceof Error ? processingError.message : 'Processing failed'
        }
      })

      throw processingError
    }

  } catch (error) {
    console.error('Upload error:', error)
    res.status(500).json({ 
      error: 'Upload failed',
      details: error instanceof Error ? error.message : 'Unknown error'
    })
  }
}

// src/pages/api/documents/status/[id].ts
import { NextApiRequest, NextApiResponse } from 'next'
import { getServerSession } from 'next-auth/next'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/prisma'
import { documentProcessor } from '@/services/pythonServices'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const session = await getServerSession(req, res, authOptions)
    if (!session) {
      return res.status(401).json({ error: 'Unauthorized' })
    }

    const { id } = req.query

    if (!id || typeof id !== 'string') {
      return res.status(400).json({ error: 'Invalid document ID' })
    }

    // 从数据库获取文档信息
    const document = await prisma.document.findFirst({
      where: {
        id,
        uploadedBy: session.user.id, // 确保用户只能查看自己的文档
      },
      include: {
        chunks: {
          select: {
            id: true,
            chunkIndex: true,
            createdAt: true,
          },
          orderBy: {
            chunkIndex: 'asc'
          }
        }
      }
    })

    if (!document) {
      return res.status(404).json({ error: 'Document not found' })
    }

    // 如果文档还在处理中，尝试从Python服务获取最新状态
    let processingStatus = null
    if (document.status === 'PROCESSING') {
      try {
        processingStatus = await documentProcessor.getProcessingStatus(document.id)
      } catch (error) {
        console.warn('Failed to get processing status from Python service:', error)
      }
    }

    res.status(200).json({
      document: {
        id: document.id,
        filename: document.filename,
        originalName: document.originalName,
        mimeType: document.mimeType,
        size: document.size,
        status: document.status,
        errorMessage: document.errorMessage,
        chunksCount: document.chunks.length,
        createdAt: document.createdAt,
        updatedAt: document.updatedAt,
        processingStartedAt: document.processingStartedAt,
        processingCompletedAt: document.processingCompletedAt,
      },
      processingStatus,
      chunks: document.chunks
    })

  } catch (error) {
    console.error('Status check error:', error)
    res.status(500).json({ error: 'Failed to get document status' })
  }
}
```

**Task 3.3.2: 文档列表和管理API**

```typescript
// src/pages/api/documents/list.ts
import { NextApiRequest, NextApiResponse } from 'next'
import { getServerSession } from 'next-auth/next'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/prisma'
import { CacheService } from '@/lib/redis'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const session = await getServerSession(req, res, authOptions)
    if (!session) {
      return res.status(401).json({ error: 'Unauthorized' })
    }

    const { page = '1', limit = '10', status, search } = req.query

    const pageNum = parseInt(page as string)
    const limitNum = parseInt(limit as string)
    const offset = (pageNum - 1) * limitNum

    // 构建查询条件
    const where: any = {
      uploadedBy: session.user.id,
    }

    if (status && typeof status === 'string') {
      where.status = status.toUpperCase()
    }

    if (search && typeof search === 'string') {
      where.OR = [
        { filename: { contains: search, mode: 'insensitive' } },
        { originalName: { contains: search, mode: 'insensitive' } },
      ]
    }

    // 检查缓存
    const cacheKey = `documents:${session.user.id}:${pageNum}:${limitNum}:${status || 'all'}:${search || ''}`
    const cachedResult = await CacheService.get(cacheKey)

    if (cachedResult) {
      return res.status(200).json(cachedResult)
    }

    // 查询文档
    const [documents, total] = await Promise.all([
      prisma.document.findMany({
        where,
        include: {
          _count: {
            select: {
              chunks: true
            }
          }
        },
        orderBy: {
          createdAt: 'desc'
        },
        skip: offset,
        take: limitNum,
      }),
      prisma.document.count({ where })
    ])

    const result = {
      documents: documents.map(doc => ({
        id: doc.id,
        filename: doc.filename,
        originalName: doc.originalName,
        mimeType: doc.mimeType,
        size: doc.size,
        status: doc.status,
        chunksCount: doc._count.chunks,
        createdAt: doc.createdAt,
        updatedAt: doc.updatedAt,
        processingStartedAt: doc.processingStartedAt,
        processingCompletedAt: doc.processingCompletedAt,
      })),
      pagination: {
        page: pageNum,
        limit: limitNum,
        total,
        totalPages: Math.ceil(total / limitNum),
        hasNext: pageNum * limitNum < total,
        hasPrev: pageNum > 1,
      }
    }

    // 缓存结果（5分钟）
    await CacheService.set(cacheKey, result, 300)

    res.status(200).json(result)

  } catch (error) {
    console.error('List documents error:', error)
    res.status(500).json({ error: 'Failed to list documents' })
  }
}

// src/pages/api/documents/delete/[id].ts
import { NextApiRequest, NextApiResponse } from 'next'
import { getServerSession } from 'next-auth/next'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/prisma'
import { CacheService } from '@/lib/redis'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'DELETE') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const session = await getServerSession(req, res, authOptions)
    if (!session) {
      return res.status(401).json({ error: 'Unauthorized' })
    }

    const { id } = req.query

    if (!id || typeof id !== 'string') {
      return res.status(400).json({ error: 'Invalid document ID' })
    }

    // 检查文档是否存在且属于当前用户
    const document = await prisma.document.findFirst({
      where: {
        id,
        uploadedBy: session.user.id,
      }
    })

    if (!document) {
      return res.status(404).json({ error: 'Document not found' })
    }

    // 软删除：标记为已删除状态
    await prisma.document.update({
      where: { id },
      data: {
        status: 'DELETED',
        updatedAt: new Date(),
      }
    })

    // 清除相关缓存
    const cachePattern = `documents:${session.user.id}:*`
    // 注意：这里需要实现模式匹配删除，简化处理可以清除用户所有文档缓存
    await CacheService.del(`documents:${session.user.id}`)

    res.status(200).json({
      message: 'Document deleted successfully',
      documentId: id
    })

  } catch (error) {
    console.error('Delete document error:', error)
    res.status(500).json({ error: 'Failed to delete document' })
  }
}
```

#### ✅ 验收标准3.3

- [ ] 文档上传API正常工作
- [ ] 文件类型和大小验证正确
- [ ] 文档状态跟踪准确
- [ ] 文档列表查询和分页正常
- [ ] 文档删除功能正常
- [ ] 缓存机制正常工作
- [ ] 与Python文档处理服务集成正常

---

### 子阶段3.4：RAG查询API开发 (2-3天) 🔍

**状态**: 🔴 待开始  
**优先级**: ⚡ 中高  
**依赖**: 子阶段3.3完成

#### 🎯 具体任务清单

**Task 3.4.1: RAG查询API**

```typescript
// src/pages/api/query/index.ts
import { NextApiRequest, NextApiResponse } from 'next'
import { getServerSession } from 'next-auth/next'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/prisma'
import { ragService } from '@/services/pythonServices'
import { CacheService } from '@/lib/redis'
import { querySchema } from '@/lib/validation'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const session = await getServerSession(req, res, authOptions)
    if (!session) {
      return res.status(401).json({ error: 'Unauthorized' })
    }

    // 验证请求数据
    const validation = querySchema.safeParse(req.body)
    if (!validation.success) {
      return res.status(400).json({
        error: 'Invalid query data',
        details: validation.error.errors
      })
    }

    const { query, topK = 5, useCache = true } = validation.data

    // 生成缓存键
    const cacheKey = `query:${Buffer.from(query).toString('base64')}:${topK}`

    // 检查缓存
    if (useCache) {
      const cachedResult = await CacheService.get(cacheKey)
      if (cachedResult) {
        return res.status(200).json({
          ...cachedResult,
          cached: true
        })
      }
    }

    // 记录查询开始
    const queryRecord = await prisma.query.create({
      data: {
        text: query,
        userId: session.user.id,
        status: 'PROCESSING',
      }
    })

    const startTime = Date.now()

    try {
      // 调用RAG服务
      const ragResult = await ragService.query(query, topK, true)
      
      const responseTime = Date.now() - startTime

      // 更新查询记录
      await prisma.query.update({
        where: { id: queryRecord.id },
        data: {
          response: ragResult.answer,
          responseTime,
          sources: ragResult.sources,
          status: 'COMPLETED',
          metadata: {
            topK,
            searchTime: ragResult.query_time,
            sourcesCount: ragResult.sources?.length || 0,
          }
        }
      })

      const result = {
        queryId: queryRecord.id,
        answer: ragResult.answer,
        sources: ragResult.sources || [],
        responseTime,
        metadata: {
          topK,
          searchTime: ragResult.query_time,
          sourcesCount: ragResult.sources?.length || 0,
        }
      }

      // 缓存结果（1小时）
      if (useCache) {
        await CacheService.set(cacheKey, result, 3600)
      }

      res.status(200).json(result)

    } catch (ragError) {
      // 更新查询记录为失败
      await prisma.query.update({
        where: { id: queryRecord.id },
        data: {
          status: 'FAILED',
          responseTime: Date.now() - startTime,
          metadata: {
            error: ragError instanceof Error ? ragError.message : 'Unknown error'
          }
        }
      })

      throw ragError
    }

  } catch (error) {
    console.error('Query error:', error)
    res.status(500).json({
      error: 'Query failed',
      details: error instanceof Error ? error.message : 'Unknown error'
    })
  }
}

// src/pages/api/query/history.ts
import { NextApiRequest, NextApiResponse } from 'next'
import { getServerSession } from 'next-auth/next'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/prisma'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const session = await getServerSession(req, res, authOptions)
    if (!session) {
      return res.status(401).json({ error: 'Unauthorized' })
    }

    const { page = '1', limit = '20', status } = req.query

    const pageNum = parseInt(page as string)
    const limitNum = parseInt(limit as string)
    const offset = (pageNum - 1) * limitNum

    const where: any = {
      userId: session.user.id,
    }

    if (status && typeof status === 'string') {
      where.status = status.toUpperCase()
    }

    const [queries, total] = await Promise.all([
      prisma.query.findMany({
        where,
        orderBy: {
          createdAt: 'desc'
        },
        skip: offset,
        take: limitNum,
        select: {
          id: true,
          text: true,
          response: true,
          responseTime: true,
          status: true,
          createdAt: true,
          metadata: true,
        }
      }),
      prisma.query.count({ where })
    ])

    res.status(200).json({
      queries,
      pagination: {
        page: pageNum,
        limit: limitNum,
        total,
        totalPages: Math.ceil(total / limitNum),
        hasNext: pageNum * limitNum < total,
        hasPrev: pageNum > 1,
      }
    })

  } catch (error) {
    console.error('Query history error:', error)
    res.status(500).json({ error: 'Failed to get query history' })
  }
}
```

**Task 3.4.2: 聊天API和实时查询**

```typescript
// src/pages/api/query/chat.ts
import { NextApiRequest, NextApiResponse } from 'next'
import { getServerSession } from 'next-auth/next'
import { authOptions } from '@/lib/auth'
import { ragService } from '@/services/pythonServices'
import { chatSchema } from '@/lib/validation'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const session = await getServerSession(req, res, authOptions)
    if (!session) {
      return res.status(401).json({ error: 'Unauthorized' })
    }

    const validation = chatSchema.safeParse(req.body)
    if (!validation.success) {
      return res.status(400).json({
        error: 'Invalid chat data',
        details: validation.error.errors
      })
    }

    const { messages } = validation.data

    // 调用RAG服务的聊天接口
    const chatResult = await ragService.chat(messages)

    res.status(200).json({
      response: chatResult.response,
      context: chatResult.context,
      sources: chatResult.sources || [],
    })

  } catch (error) {
    console.error('Chat error:', error)
    res.status(500).json({
      error: 'Chat failed',
      details: error instanceof Error ? error.message : 'Unknown error'
    })
  }
}

// src/pages/api/documents/[id]/qa.ts
import { NextApiRequest, NextApiResponse } from 'next'
import { getServerSession } from 'next-auth/next'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/prisma'
import { ragService } from '@/services/pythonServices'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const session = await getServerSession(req, res, authOptions)
    if (!session) {
      return res.status(401).json({ error: 'Unauthorized' })
    }

    const { id } = req.query
    const { question } = req.body

    if (!id || typeof id !== 'string') {
      return res.status(400).json({ error: 'Invalid document ID' })
    }

    if (!question || typeof question !== 'string') {
      return res.status(400).json({ error: 'Question is required' })
    }

    // 验证文档所有权
    const document = await prisma.document.findFirst({
      where: {
        id,
        uploadedBy: session.user.id,
        status: 'COMPLETED', // 只有处理完成的文档才能查询
      }
    })

    if (!document) {
      return res.status(404).json({ error: 'Document not found or not processed' })
    }

    // 调用RAG服务针对特定文档的QA
    const qaResult = await ragService.documentQA(id, question)

    res.status(200).json({
      documentId: id,
      question,
      answer: qaResult.answer,
      sources: qaResult.sources || [],
      confidence: qaResult.confidence,
    })

  } catch (error) {
    console.error('Document QA error:', error)
    res.status(500).json({
      error: 'Document QA failed',
      details: error instanceof Error ? error.message : 'Unknown error'
    })
  }
}
```

#### ✅ 验收标准3.4

- [ ] RAG查询API正常工作
- [ ] 查询历史记录正确保存
- [ ] 缓存机制正确实现
- [ ] 聊天接口功能正常
- [ ] 针对特定文档的QA功能正常
- [ ] 响应时间和性能满足要求
- [ ] 与Python RAG服务集成正常

---

### 子阶段3.5：管理和监控API (2-3天) 📊

**状态**: 🔴 待开始  
**优先级**: ⚡ 中等  
**依赖**: 子阶段3.4完成

#### 🎯 具体任务清单

**Task 3.5.1: 系统健康监控API**

```typescript
// src/pages/api/health.ts
import { NextApiRequest, NextApiResponse } from 'next'
import { prisma } from '@/lib/prisma'
import { redis } from '@/lib/redis'
import { documentProcessor, vectorService, ragService } from '@/services/pythonServices'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
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

// src/pages/api/admin/system.ts
import { NextApiRequest, NextApiResponse } from 'next'
import { getServerSession } from 'next-auth/next'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/prisma'
import { redis } from '@/lib/redis'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const session = await getServerSession(req, res, authOptions)
    if (!session || session.user.role !== 'ADMIN') {
      return res.status(403).json({ error: 'Admin access required' })
    }

    // 获取系统统计信息
    const [
      userCount,
      documentCount,
      queryCount,
      documentStats,
      queryStats,
    ] = await Promise.all([
      prisma.user.count(),
      prisma.document.count(),
      prisma.query.count(),
      prisma.document.groupBy({
        by: ['status'],
        _count: true,
      }),
      prisma.query.groupBy({
        by: ['status'],
        _count: true,
      }),
    ])

    // 获取最近24小时的活动
    const last24Hours = new Date(Date.now() - 24 * 60 * 60 * 1000)
    const recentActivity = await Promise.all([
      prisma.document.count({
        where: { createdAt: { gte: last24Hours } }
      }),
      prisma.query.count({
        where: { createdAt: { gte: last24Hours } }
      }),
    ])

    // Redis内存使用情况
    let redisInfo = null
    try {
      const info = await redis.info('memory')
      redisInfo = {
        usedMemory: info.match(/used_memory:(\d+)/)?.[1],
        usedMemoryHuman: info.match(/used_memory_human:(.+)/)?.[1],
      }
    } catch (error) {
      console.warn('Failed to get Redis info:', error)
    }

    res.status(200).json({
      overview: {
        totalUsers: userCount,
        totalDocuments: documentCount,
        totalQueries: queryCount,
        last24Hours: {
          newDocuments: recentActivity[0],
          newQueries: recentActivity[1],
        }
      },
      documentStats: documentStats.reduce((acc, stat) => {
        acc[stat.status] = stat._count
        return acc
      }, {} as Record<string, number>),
      queryStats: queryStats.reduce((acc, stat) => {
        acc[stat.status] = stat._count
        return acc
      }, {} as Record<string, number>),
      redis: redisInfo,
      timestamp: new Date().toISOString(),
    })

  } catch (error) {
    console.error('System stats error:', error)
    res.status(500).json({ error: 'Failed to get system statistics' })
  }
}
```

**Task 3.5.2: 用户管理API**

```typescript
// src/pages/api/admin/users.ts
import { NextApiRequest, NextApiResponse } from 'next'
import { getServerSession } from 'next-auth/next'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/prisma'

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const session = await getServerSession(req, res, authOptions)
  if (!session || session.user.role !== 'ADMIN') {
    return res.status(403).json({ error: 'Admin access required' })
  }

  switch (req.method) {
    case 'GET':
      return handleGetUsers(req, res)
    case 'PUT':
      return handleUpdateUser(req, res)
    case 'DELETE':
      return handleDeleteUser(req, res)
    default:
      return res.status(405).json({ error: 'Method not allowed' })
  }
}

async function handleGetUsers(req: NextApiRequest, res: NextApiResponse) {
  try {
    const { page = '1', limit = '20', search, role, status } = req.query

    const pageNum = parseInt(page as string)
    const limitNum = parseInt(limit as string)
    const offset = (pageNum - 1) * limitNum

    const where: any = {}

    if (search && typeof search === 'string') {
      where.OR = [
        { email: { contains: search, mode: 'insensitive' } },
        { name: { contains: search, mode: 'insensitive' } },
      ]
    }

    if (role && typeof role === 'string') {
      where.role = role.toUpperCase()
    }

    if (status && typeof status === 'string') {
      where.isActive = status === 'active'
    }

    const [users, total] = await Promise.all([
      prisma.user.findMany({
        where,
        select: {
          id: true,
          email: true,
          name: true,
          role: true,
          isActive: true,
          createdAt: true,
          updatedAt: true,
          _count: {
            select: {
              documents: true,
              queries: true,
            }
          }
        },
        orderBy: {
          createdAt: 'desc'
        },
        skip: offset,
        take: limitNum,
      }),
      prisma.user.count({ where })
    ])

    res.status(200).json({
      users,
      pagination: {
        page: pageNum,
        limit: limitNum,
        total,
        totalPages: Math.ceil(total / limitNum),
        hasNext: pageNum * limitNum < total,
        hasPrev: pageNum > 1,
      }
    })

  } catch (error) {
    console.error('Get users error:', error)
    res.status(500).json({ error: 'Failed to get users' })
  }
}

async function handleUpdateUser(req: NextApiRequest, res: NextApiResponse) {
  try {
    const { userId, role, isActive } = req.body

    if (!userId) {
      return res.status(400).json({ error: 'User ID is required' })
    }

    const updateData: any = {}
    if (role !== undefined) updateData.role = role
    if (isActive !== undefined) updateData.isActive = isActive

    const updatedUser = await prisma.user.update({
      where: { id: userId },
      data: updateData,
      select: {
        id: true,
        email: true,
        name: true,
        role: true,
        isActive: true,
        updatedAt: true,
      }
    })

    res.status(200).json({
      message: 'User updated successfully',
      user: updatedUser
    })

  } catch (error) {
    console.error('Update user error:', error)
    res.status(500).json({ error: 'Failed to update user' })
  }
}

async function handleDeleteUser(req: NextApiRequest, res: NextApiResponse) {
  try {
    const { userId } = req.body

    if (!userId) {
      return res.status(400).json({ error: 'User ID is required' })
    }

    // 软删除：设置为非活跃状态
    await prisma.user.update({
      where: { id: userId },
      data: { isActive: false }
    })

    res.status(200).json({
      message: 'User deactivated successfully'
    })

  } catch (error) {
    console.error('Delete user error:', error)
    res.status(500).json({ error: 'Failed to delete user' })
  }
}
```

#### ✅ 验收标准3.5

- [ ] 系统健康检查API正常工作
- [ ] 系统统计信息准确显示
- [ ] 用户管理功能完整
- [ ] 管理员权限验证正确
- [ ] 监控数据格式标准化
- [ ] 错误处理机制完善

---

## 📊 执行进度跟踪

### 进度检查点

- [ ] **Day 4**: 子阶段3.1完成，项目基础设施就绪
- [ ] **Day 8**: 子阶段3.2完成，核心服务集成完成
- [ ] **Day 12**: 子阶段3.3完成，文档管理API就绪
- [ ] **Day 14**: 子阶段3.4完成，RAG查询API就绪
- [ ] **Day 15**: 子阶段3.5完成，管理监控API就绪

### 风险缓解计划

1. **数据库迁移风险**: 使用Prisma Migration确保数据库变更可控
2. **Python服务集成风险**: 实现降级机制，确保后端服务独立可用
3. **性能风险**: 实现多层缓存和连接池优化
4. **安全风险**: 严格的认证授权和数据验证

---

## 🔧 技术规范

### API设计规范

```typescript
// 统一响应格式
interface ApiResponse<T = any> {
  success: boolean
  data?: T
  error?: string
  message?: string
  timestamp: string
  requestId?: string
}

// 分页响应格式
interface PaginatedResponse<T> extends ApiResponse<T[]> {
  pagination: {
    page: number
    limit: number
    total: number
    totalPages: number
    hasNext: boolean
    hasPrev: boolean
  }
}

// 错误响应格式
interface ErrorResponse {
  success: false
  error: string
  details?: any
  code?: string
  timestamp: string
}
```

### 数据验证规范

```typescript
// src/lib/validation.ts
import { z } from 'zod'

export const uploadSchema = z.object({
  filename: z.string().min(1).max(255),
  size: z.number().min(1).max(10 * 1024 * 1024), // 10MB
  mimeType: z.string(),
})

export const querySchema = z.object({
  query: z.string().min(1).max(1000),
  topK: z.number().min(1).max(20).optional(),
  useCache: z.boolean().optional(),
})

export const chatSchema = z.object({
  messages: z.array(z.object({
    role: z.enum(['user', 'assistant']),
    content: z.string().min(1),
  })).min(1).max(50),
})

export const registerSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8).max(128),
  name: z.string().min(1).max(100).optional(),
})
```

### 错误处理规范

```typescript
// src/utils/errors.ts
export class AppError extends Error {
  constructor(
    public message: string,
    public statusCode: number = 500,
    public code?: string
  ) {
    super(message)
    this.name = 'AppError'
  }
}

export class ValidationError extends AppError {
  constructor(message: string, details?: any) {
    super(message, 400, 'VALIDATION_ERROR')
    this.details = details
  }
}

export class AuthError extends AppError {
  constructor(message: string = 'Unauthorized') {
    super(message, 401, 'AUTH_ERROR')
  }
}

export class NotFoundError extends AppError {
  constructor(message: string = 'Resource not found') {
    super(message, 404, 'NOT_FOUND')
  }
}

// 错误处理中间件
export function errorHandler(error: Error, req: NextApiRequest, res: NextApiResponse) {
  console.error('API Error:', error)

  if (error instanceof AppError) {
    return res.status(error.statusCode).json({
      success: false,
      error: error.message,
      code: error.code,
      timestamp: new Date().toISOString(),
    })
  }

  return res.status(500).json({
    success: false,
    error: 'Internal server error',
    timestamp: new Date().toISOString(),
  })
}
```

---

## ✅ 最终交付清单

### 代码交付

- [ ] 完整的Next.js后端项目
- [ ] 所有API端点实现
- [ ] 数据库Schema和Migration
- [ ] TypeScript类型定义
- [ ] 中间件和工具函数

### 配置交付

- [ ] Docker配置文件
- [ ] 环境变量配置
- [ ] 数据库配置
- [ ] 缓存配置

### 文档交付

- [ ] API接口文档
- [ ] 数据库设计文档
- [ ] 部署指南
- [ ] 开发指南

### 测试交付

- [ ] API集成测试
- [ ] 数据库操作测试
- [ ] Python服务集成测试
- [ ] 性能测试报告

---

**执行原则**:

1. **API优先设计** - 先设计API接口，再实现具体功能
2. **安全第一** - 每个接口都要有适当的认证和授权
3. **性能优化** - 合理使用缓存和数据库优化
4. **错误处理** - 完善的错误处理和日志记录
5. **测试驱动** - 每个功能都要有对应的测试

此文档将作为阶段3执行的最高指导，所有任务都应该按照此文档的规范和标准进行。
