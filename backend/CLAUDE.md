# RAG后端API系统 - 开发指导文档

## 📋 服务概览

**RAG Backend** 是基于Next.js 14的全栈后端服务，提供完整的文档管理、用户认证、查询处理和AI问答API。采用现代TypeScript架构，集成Prisma ORM和PostgreSQL数据库，支持JWT认证和跨域访问。

### 🏗️ 技术栈
- **框架**: Next.js 14 + TypeScript + API Routes
- **数据库**: PostgreSQL + Prisma ORM
- **认证**: JWT + bcryptjs
- **缓存**: Redis (ioredis)
- **消息队列**: RabbitMQ (amqplib)
- **文件处理**: Formidable + Sharp
- **HTTP客户端**: Axios
- **端口**: 3001

### 📁 项目结构
```
backend/
├── src/
│   ├── lib/                 # 核心库和中间件
│   │   ├── jwtAuth.ts      # JWT认证中间件
│   │   ├── cors.ts         # CORS统一处理
│   │   └── prisma.ts       # Prisma客户端
│   ├── pages/api/          # API路由
│   │   ├── auth/           # 认证相关API
│   │   ├── documents/      # 文档管理API
│   │   └── query/          # AI查询API
│   └── types/              # TypeScript类型定义
├── prisma/
│   ├── schema.prisma       # 数据库模式
│   └── migrations/         # 数据库迁移
└── tests/                  # 测试用例
```

## 🎯 核心功能模块

### 🔐 1. 认证系统 (`/api/auth/`)
- **用户注册**: `POST /api/auth/signup`
- **用户登录**: `POST /api/auth/signin`
- **用户信息**: `GET /api/auth/user`
- **JWT令牌**: 7天有效期，自动续签机制

### 📄 2. 文档管理 (`/api/documents/`)
- **文档上传**: `POST /api/documents/upload`
- **文档列表**: `GET /api/documents/list` (分页+搜索)
- **文档状态**: `GET /api/documents/status/[id]`
- **文档删除**: `DELETE /api/documents/delete/[id]`

### 🤖 3. AI查询系统 (`/api/query/`)
- **智能问答**: `POST /api/query`
- **查询历史**: `GET /api/query/history`
- **RAG集成**: 调用Python服务实现检索增强生成

## 🧪 开发和调试指南

### 环境配置
```bash
# 必需环境变量
DATABASE_URL="postgresql://postgres:password@localhost:5432/rag_db"
REDIS_URL="redis://localhost:6379"
RABBITMQ_URL="amqp://guest:guest@localhost:5672"
JWT_SECRET="your-super-secure-jwt-secret-key"

# Python服务端点
DOCUMENT_PROCESSOR_URL="http://document-processor:8001"
VECTOR_SERVICE_URL="http://vector-service:8002"
RAG_SERVICE_URL="http://rag-service:8003"

# 文件上传配置
MAX_FILE_SIZE="50MB"
UPLOAD_DIRECTORY="./uploads"
```

### 本地开发启动
```bash
# 1. 安装依赖
npm install

# 2. 生成Prisma客户端
npm run db:generate

# 3. 运行数据库迁移
npm run db:migrate

# 4. 启动开发服务器
PORT=3001 npm run dev

# 服务地址: http://localhost:3001
```

### 数据库操作
```bash
# 查看数据库状态
npx prisma studio

# 重置数据库
npx prisma migrate reset

# 推送模式变更
npx prisma db push

# 生成迁移文件
npx prisma migrate dev --name "migration_name"
```

## 🔧 核心组件详解

### JWT认证中间件
```typescript
// src/lib/jwtAuth.ts - 统一认证处理
export const withAuth = (handler: NextApiHandler): NextApiHandler => {
  return async (req: NextApiRequest, res: NextApiResponse) => {
    try {
      // 1. 提取JWT令牌
      const authHeader = req.headers.authorization;
      if (!authHeader || !authHeader.startsWith('Bearer ')) {
        return res.status(401).json({ error: 'No token provided' });
      }

      // 2. 验证令牌
      const token = authHeader.substring(7);
      const decoded = jwt.verify(token, JWT_SECRET!) as JwtPayload;
      
      // 3. 注入用户信息
      req.user = { 
        id: decoded.userId, 
        email: decoded.email,
        role: decoded.role 
      };
      
      return handler(req, res);
    } catch (error) {
      return res.status(401).json({ error: 'Invalid or expired token' });
    }
  };
};
```

### CORS统一处理
```typescript
// src/lib/cors.ts - 跨域请求处理
export const corsMiddleware = (handler: NextApiHandler): NextApiHandler => {
  return async (req: NextApiRequest, res: NextApiResponse) => {
    // 设置CORS头
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Authorization, Content-Type');
    res.setHeader('Access-Control-Max-Age', '86400');

    // 处理OPTIONS预检请求
    if (req.method === 'OPTIONS') {
      return res.status(200).end();
    }

    return handler(req, res);
  };
};
```

### Prisma数据库客户端
```typescript
// src/lib/prisma.ts - 数据库连接
import { PrismaClient } from '@prisma/client';

declare global {
  var prisma: PrismaClient | undefined;
}

export const prisma = globalThis.prisma ?? new PrismaClient({
  log: ['query', 'error', 'warn'],
  errorFormat: 'pretty',
});

if (process.env.NODE_ENV !== 'production') {
  globalThis.prisma = prisma;
}
```

## 📊 API端点详细规范

### 认证API

#### 1. 用户注册
```typescript
// POST /api/auth/signup
interface SignupRequest {
  email: string;
  password: string;
  name?: string;
}

interface SignupResponse {
  success: boolean;
  user: {
    id: string;
    email: string;
    name: string;
    role: UserRole;
  };
  token: string;
}
```

#### 2. 用户登录
```typescript
// POST /api/auth/signin  
interface SigninRequest {
  email: string;
  password: string;
}

interface SigninResponse {
  success: boolean;
  user: {
    id: string;
    email: string;
    name: string;
    role: UserRole;
  };
  token: string;
}
```

### 文档管理API

#### 1. 文档上传
```typescript
// POST /api/documents/upload
// Content-Type: multipart/form-data
interface UploadRequest {
  file: File; // 支持PDF, DOCX, TXT, MD
}

interface UploadResponse {
  success: boolean;
  document: {
    id: string;
    originalName: string;
    filename: string;
    mimeType: string;
    size: number;
    status: DocumentStatus;
  };
  processingStarted: boolean;
}
```

#### 2. 文档列表查询
```typescript
// GET /api/documents/list
interface ListQuery {
  page?: number;     // 页码，默认1
  limit?: number;    // 每页数量，默认10，最大100
  search?: string;   // 搜索关键词
  status?: DocumentStatus; // 状态筛选
}

interface ListResponse {
  success: boolean;
  documents: Array<{
    id: string;
    originalName: string;
    size: number;
    status: DocumentStatus;
    chunksCount: number; // 通过 _count.chunks 计算
    createdAt: string;
  }>;
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNext: boolean;
    hasPrev: boolean;
  };
}
```

### AI查询API

#### 1. 智能问答
```typescript
// POST /api/query
interface QueryRequest {
  query: string;
  top_k?: number; // 检索文档数量，默认3
}

interface QueryResponse {
  success: boolean;
  answer: string;
  sources: Array<{
    documentId: string;
    content: string;
    score: number;
    metadata?: Record<string, any>;
  }>;
  queryTime: number; // 毫秒
  tokensUsed?: number;
}
```

#### 2. 查询历史
```typescript
// GET /api/query/history
interface HistoryQuery {
  limit?: number;   // 限制数量，默认20
  offset?: number;  // 偏移量，默认0
}

interface HistoryResponse {
  success: boolean;
  queries: Array<{
    id: string;
    text: string;
    response: string;
    responseTime: number;
    status: QueryStatus;
    createdAt: string;
  }>;
  total: number;
}
```

## 🔄 服务集成

### Python服务客户端
```typescript
// 文档处理服务客户端
export class DocumentProcessorClient {
  private baseURL = process.env.DOCUMENT_PROCESSOR_URL!;
  
  async processDocument(file: Buffer, filename: string, documentId: string) {
    const formData = new FormData();
    formData.append('file', new Blob([file]), filename);
    formData.append('document_id', documentId);
    formData.append('enable_chunking', 'true');
    
    const response = await axios.post(`${this.baseURL}/process-document`, formData);
    return response.data;
  }
  
  async getProcessingStatus(documentId: string) {
    const response = await axios.get(`${this.baseURL}/processing-status/${documentId}`);
    return response.data;
  }
}

// RAG服务客户端
export class RAGServiceClient {
  private baseURL = process.env.RAG_SERVICE_URL!;
  
  async query(queryText: string, topK: number = 3) {
    const response = await axios.post(`${this.baseURL}/query`, {
      query: queryText,
      top_k: topK
    });
    return response.data;
  }
}
```

### Redis缓存策略
```typescript
// 缓存实现
import Redis from 'ioredis';

export class CacheService {
  private redis = new Redis(process.env.REDIS_URL!);
  
  // 文档列表缓存 - 5分钟
  async cacheDocumentList(userId: string, params: any, data: any) {
    const key = `docs:${userId}:${JSON.stringify(params)}`;
    await this.redis.setex(key, 300, JSON.stringify(data));
  }
  
  // 查询结果缓存 - 1小时
  async cacheQueryResult(queryHash: string, result: any) {
    const key = `query:${queryHash}`;
    await this.redis.setex(key, 3600, JSON.stringify(result));
  }
  
  // 用户会话缓存
  async cacheUserSession(userId: string, sessionData: any) {
    const key = `session:${userId}`;
    await this.redis.setex(key, 604800, JSON.stringify(sessionData)); // 7天
  }
}
```

## 🚨 错误处理与日志

### 统一错误处理
```typescript
// 错误处理中间件
export const withErrorHandling = (handler: NextApiHandler): NextApiHandler => {
  return async (req: NextApiRequest, res: NextApiResponse) => {
    try {
      return await handler(req, res);
    } catch (error) {
      console.error('API Error:', error);
      
      // Prisma错误处理
      if (error instanceof Prisma.PrismaClientKnownRequestError) {
        if (error.code === 'P2002') {
          return res.status(409).json({ error: 'Unique constraint violation' });
        }
        if (error.code === 'P2025') {
          return res.status(404).json({ error: 'Record not found' });
        }
      }
      
      // JWT错误处理
      if (error instanceof jwt.JsonWebTokenError) {
        return res.status(401).json({ error: 'Invalid token' });
      }
      
      // 默认错误处理
      return res.status(500).json({ 
        error: 'Internal server error',
        message: process.env.NODE_ENV === 'development' ? error.message : undefined
      });
    }
  };
};
```

### 结构化日志
```typescript
// 日志工具
export class Logger {
  static info(message: string, meta?: object) {
    console.log(JSON.stringify({
      level: 'info',
      timestamp: new Date().toISOString(),
      message,
      ...meta
    }));
  }
  
  static error(message: string, error?: Error, meta?: object) {
    console.error(JSON.stringify({
      level: 'error',
      timestamp: new Date().toISOString(),
      message,
      error: error?.message,
      stack: error?.stack,
      ...meta
    }));
  }
  
  static apiCall(req: NextApiRequest, res: NextApiResponse, duration: number) {
    Logger.info('API Call', {
      method: req.method,
      url: req.url,
      status: res.statusCode,
      duration: `${duration}ms`,
      userAgent: req.headers['user-agent']
    });
  }
}
```

## 🧪 测试策略

### 单元测试
```typescript
// tests/auth.test.ts
import { createMocks } from 'node-mocks-http';
import handler from '../src/pages/api/auth/signin';

describe('/api/auth/signin', () => {
  it('should return 200 and token for valid credentials', async () => {
    const { req, res } = createMocks({
      method: 'POST',
      body: {
        email: 'test@example.com',
        password: 'testpassword123'
      }
    });

    await handler(req, res);

    expect(res._getStatusCode()).toBe(200);
    const data = JSON.parse(res._getData());
    expect(data.success).toBe(true);
    expect(data.token).toBeDefined();
  });
});
```

### 集成测试
```typescript
// tests/documents.integration.test.ts
describe('Document Management Flow', () => {
  let authToken: string;

  beforeAll(async () => {
    // 登录获取token
    const loginResponse = await request(app)
      .post('/api/auth/signin')
      .send({ email: 'test@example.com', password: 'testpassword123' });
    authToken = loginResponse.body.token;
  });

  it('should upload document successfully', async () => {
    const response = await request(app)
      .post('/api/documents/upload')
      .set('Authorization', `Bearer ${authToken}`)
      .attach('file', Buffer.from('test content'), 'test.txt');

    expect(response.status).toBe(200);
    expect(response.body.success).toBe(true);
  });
});
```

## 📈 性能优化

### 数据库优化
```sql
-- 重要索引
CREATE INDEX CONCURRENTLY idx_documents_user_status 
ON documents(uploaded_by, status);

CREATE INDEX CONCURRENTLY idx_documents_created_at 
ON documents(created_at DESC);

CREATE INDEX CONCURRENTLY idx_queries_user_created 
ON queries(user_id, created_at DESC);

-- 全文搜索索引
CREATE INDEX CONCURRENTLY idx_documents_name_gin 
ON documents USING gin(to_tsvector('english', original_name));
```

### 缓存策略
```typescript
// 智能缓存键生成
export const generateCacheKey = (prefix: string, ...params: any[]) => {
  const hash = crypto.createHash('md5')
    .update(JSON.stringify(params))
    .digest('hex');
  return `${prefix}:${hash}`;
};

// 缓存装饰器
export const withCache = (ttl: number) => {
  return (target: any, propertyKey: string, descriptor: PropertyDescriptor) => {
    const originalMethod = descriptor.value;
    
    descriptor.value = async function (...args: any[]) {
      const cacheKey = generateCacheKey(propertyKey, ...args);
      const cached = await redis.get(cacheKey);
      
      if (cached) {
        return JSON.parse(cached);
      }
      
      const result = await originalMethod.apply(this, args);
      await redis.setex(cacheKey, ttl, JSON.stringify(result));
      
      return result;
    };
  };
};
```

## 🔒 安全措施

### 输入验证
```typescript
import { z } from 'zod';

// 查询参数验证
export const QuerySchema = z.object({
  query: z.string().min(1).max(1000),
  top_k: z.number().int().min(1).max(20).optional()
});

// 文档上传验证
export const validateUploadedFile = (file: any) => {
  const allowedTypes = ['application/pdf', 'text/plain', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
  const maxSize = 50 * 1024 * 1024; // 50MB
  
  if (!allowedTypes.includes(file.mimetype)) {
    throw new Error('Unsupported file type');
  }
  
  if (file.size > maxSize) {
    throw new Error('File size exceeds limit');
  }
  
  return true;
};
```

### 速率限制
```typescript
// 简单内存速率限制器
export class RateLimiter {
  private requests = new Map<string, number[]>();
  
  isAllowed(identifier: string, limit: number, windowMs: number): boolean {
    const now = Date.now();
    const requests = this.requests.get(identifier) || [];
    
    // 清理过期请求
    const validRequests = requests.filter(time => now - time < windowMs);
    
    if (validRequests.length >= limit) {
      return false;
    }
    
    validRequests.push(now);
    this.requests.set(identifier, validRequests);
    return true;
  }
}

// 使用示例
export const withRateLimit = (limit: number, windowMs: number) => {
  const limiter = new RateLimiter();
  
  return (handler: NextApiHandler): NextApiHandler => {
    return async (req: NextApiRequest, res: NextApiResponse) => {
      const identifier = req.headers['x-forwarded-for'] as string || req.connection.remoteAddress!;
      
      if (!limiter.isAllowed(identifier, limit, windowMs)) {
        return res.status(429).json({ error: 'Rate limit exceeded' });
      }
      
      return handler(req, res);
    };
  };
};
```

## 🚀 部署配置

### 生产环境变量
```bash
# 数据库配置
DATABASE_URL="postgresql://user:pass@prod-db:5432/rag_db"
REDIS_URL="redis://prod-redis:6379"

# 安全配置
JWT_SECRET="production-jwt-secret-minimum-32-chars"
BCRYPT_ROUNDS=12

# 服务端点
DOCUMENT_PROCESSOR_URL="http://document-processor:8001"
VECTOR_SERVICE_URL="http://vector-service:8002"  
RAG_SERVICE_URL="http://rag-service:8003"

# 性能配置
NODE_ENV=production
PORT=3001
```

### Docker配置
```dockerfile
# Dockerfile
FROM node:18-alpine AS base
RUN apk add --no-cache libc6-compat
WORKDIR /app

FROM base AS deps
COPY package*.json ./
RUN npm ci --only=production

FROM base AS builder
COPY . .
COPY --from=deps /app/node_modules ./node_modules
RUN npm run build && npm run db:generate

FROM base AS runner
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3001
CMD ["node", "server.js"]
```

## 📋 开发检查清单

### ✅ API开发规范
- [ ] 统一使用TypeScript类型
- [ ] 实现CORS中间件
- [ ] 添加JWT认证保护
- [ ] 实现输入参数验证
- [ ] 添加错误处理机制
- [ ] 实现结构化日志记录

### ✅ 数据库操作
- [ ] 使用Prisma ORM查询
- [ ] 实现事务处理
- [ ] 添加数据库索引优化
- [ ] 实现软删除机制
- [ ] 添加数据迁移脚本

### ✅ 性能优化
- [ ] 实现Redis缓存
- [ ] 添加查询结果缓存
- [ ] 实现分页查询
- [ ] 优化数据库查询
- [ ] 添加响应时间监控

### ✅ 安全措施
- [ ] 实现JWT令牌验证
- [ ] 添加请求速率限制
- [ ] 实现文件上传安全检查
- [ ] 添加SQL注入防护
- [ ] 实现XSS防护

## 🔗 相关文档

- [前端集成文档](../frontend/CLAUDE.md)
- [Python服务文档](../python-services/CLAUDE.md)
- [数据库模式文档](./prisma/schema.prisma)
- [API测试集合](./tests/)
- [部署配置](../docker-compose.yml)

---

## 🚀 **RAG系统准确率优化方案**
**重要**: Backend作为API网关，正在实施[RAG系统准确率优化方案](../CLAUDE.md#rag系统准确率优化方案)，目标将准确率从53.4%-65%提升至90-95%。

## 📅 最新更新 (2025-08-14)

### ✅ 系统重启和权限问题修复完成
- **服务重启**: 重启后成功恢复后端API服务运行
- **权限修复**: 解决.next目录权限问题，采用备份策略
- **端口配置**: 确认后端服务运行在端口3001
- **健康检查**: API健康检查正常，服务状态良好

### 🔧 重启后修复事项
1. **Next.js权限问题** (`.next目录`)
   - 问题: EACCES权限错误阻止服务启动
   - 解决: 使用`mv .next .next.bak`备份策略
   - 效果: 服务成功启动并正常运行

2. **服务端口确认** (`package.json`)
   - 开发模式: `PORT=3001 npm run dev`
   - 生产模式: `npm start` (需要先构建)
   - 健康检查: `/api/health` 端点正常响应

3. **API功能验证**
   - 认证系统: JWT认证正常
   - 文档管理: API端点可访问
   - RAG集成: 与Python服务通信正常

### 📊 当前服务状态 (2025-08-14)
- **运行端口**: 3001
- **服务状态**: ✅ 正常运行
- **健康检查**: 207状态（部分服务降级但核心功能正常）
- **主要功能**: 认证、文档管理、RAG查询均正常

### 📋 Backend优化重点

#### ✅ Phase 1: API集成优化 (进行中)
**查询历史分析** - 支持优化评估
```typescript
// src/pages/api/query/analytics.ts
export interface QueryAnalytics {
  accuracy_trend: { date: string; accuracy: number }[]
  response_time_trend: { date: string; avg_time: number }[]
  query_types: { type: string; count: number; avg_accuracy: number }[]
  failure_patterns: { pattern: string; frequency: number }[]
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === 'GET') {
    const analytics = await calculateQueryAnalytics()
    res.status(200).json(analytics)
  }
}
```

**A/B测试框架** - 优化效果验证
```typescript
// src/lib/abtest.ts
export class ABTestManager {
  async routeQuery(query: string, userId: string): Promise<'control' | 'optimized'> {
    const userGroup = await this.getUserGroup(userId)
    
    if (userGroup === 'optimized') {
      // 使用优化后的RAG服务
      return 'optimized'
    } else {
      // 使用原始RAG服务
      return 'control'
    }
  }
  
  async recordResult(queryId: string, group: string, accuracy: number, responseTime: number) {
    await prisma.abTestResult.create({
      data: { queryId, group, accuracy, responseTime }
    })
  }
}
```

#### 🔄 Phase 2: 智能路由优化 (计划中)
**查询类型识别** - 智能分发
```typescript
export class QueryRouter {
  async routeByType(query: string): Promise<string> {
    const queryType = await this.classifyQuery(query)
    
    const routes = {
      'factual': '/api/rag/factual',
      'numerical': '/api/rag/numerical', 
      'conceptual': '/api/rag/conceptual',
      'comparison': '/api/rag/comparison'
    }
    
    return routes[queryType] || '/api/rag/general'
  }
}
```

**缓存策略优化** - 响应时间提升
```typescript
export class IntelligentCache {
  async getCachedResponse(query: string): Promise<CachedResponse | null> {
    // 语义相似度缓存匹配
    const similarQueries = await this.findSimilarQueries(query, 0.9)
    
    if (similarQueries.length > 0) {
      return this.getCacheEntry(similarQueries[0].id)
    }
    
    return null
  }
}
```

#### 📈 Phase 3: 监控和反馈 (规划中)
**实时质量监控** - 准确率追踪
```typescript
export class QualityMonitor {
  async trackQueryQuality(queryId: string, feedback: UserFeedback) {
    await prisma.queryFeedback.create({
      data: {
        queryId,
        rating: feedback.rating,
        feedback_type: feedback.type,
        accuracy_score: this.calculateAccuracy(feedback)
      }
    })
    
    // 触发自动调优
    if (feedback.rating < 3) {
      await this.triggerRetraining(queryId)
    }
  }
}
```

### 🔧 开发优先级
1. **立即实施**: 查询历史分析和A/B测试框架
2. **本周实施**: 智能查询路由和缓存优化
3. **下周实施**: 实时质量监控集成
4. **月内完成**: 用户反馈循环和自动调优

### 📊 预期效果
- **Phase 1完成**: A/B测试框架支持优化验证
- **Phase 2完成**: 查询路由准确率提升，响应时间减少30%
- **Phase 3完成**: 实时监控和自动调优，整体准确率提升至90%+

详细技术方案和实施路线图请参考[根目录RAG优化方案](../CLAUDE.md#rag系统准确率优化方案)。

---

---

## 🚀 重启后快速启动指南 (2025-08-14)

### 启动后端服务
```bash
cd /home/bikeread/dev/rag_search/backend

# 如遇.next权限问题，先备份旧目录
mv .next .next.bak 2>/dev/null || true

# 启动开发服务器在端口3001
PORT=3001 npm run dev
```

### 验证服务状态
```bash
# 检查服务运行状态
curl http://localhost:3001/api/health

# 验证端口占用
lsof -i :3001
```

### 常见问题解决
1. **权限错误**: 使用`mv .next .next.bak`备份旧目录
2. **端口占用**: 使用`lsof -i :3001`检查并终止占用进程
3. **依赖问题**: 运行`npm install`重新安装依赖

---

**维护说明**: 本文档是RAG后端系统的核心开发指导，包含完整的API规范、开发流程和最佳实践。配合[前端集成文档](../frontend/CLAUDE.md)和[Python服务文档](../python-services/CLAUDE.md)，确保Backend在RAG准确率优化过程中的协调作用得到充分发挥。