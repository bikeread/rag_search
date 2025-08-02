# RAG系统全栈重构指导文档

## 重构概述

本文档指导将现有Python RAG系统重构为**React前端 + Next.js后端 + Python AI服务**的现代化全栈架构，实现前后端分离、微服务化，提升用户体验和系统可扩展性。

### 重构目标
- ✅ **现代化前端**：React + Ant Design提供优秀用户体验
- ✅ **高性能后端**：Next.js处理业务逻辑和API编排
- ✅ **专业化AI服务**：Python专注文档处理和语义检索
- ✅ **保持现有向量存储**：继续使用Milvus向量数据库
- ✅ **阶段性交付**：快速迭代，每阶段可独立运行

## 重构架构设计

### 目标架构图
```mermaid
graph TB
    subgraph "前端层 (React)"
        FE[React + Ant Design]
        FE_COMPONENTS[文档上传/聊天界面/管理面板]
    end
    
    subgraph "Next.js后端层" 
        NEXTJS[Next.js API Routes]
        AUTH[JWT认证]
        BIZ[业务逻辑编排]
        CACHE[Redis缓存]
    end
    
    subgraph "Python AI服务层"
        DOC_SVC[文档处理服务:8001]
        VECTOR_SVC[向量化服务:8002] 
        RAG_SVC[RAG检索服务:8003]
    end
    
    subgraph "存储层"
        MILVUS[Milvus向量数据库]
        POSTGRES[PostgreSQL业务数据]
        REDIS[Redis缓存]
        RABBITMQ[RabbitMQ消息队列]
    end
    
    FE --> NEXTJS
    NEXTJS --> DOC_SVC
    NEXTJS --> VECTOR_SVC  
    NEXTJS --> RAG_SVC
    NEXTJS --> POSTGRES
    NEXTJS --> REDIS
    
    DOC_SVC --> RABBITMQ
    VECTOR_SVC --> MILVUS
    RAG_SVC --> MILVUS
    
    style FE fill:#e1f5fe
    style NEXTJS fill:#f3e5f5
    style DOC_SVC fill:#e8f5e8
    style VECTOR_SVC fill:#e8f5e8
    style RAG_SVC fill:#e8f5e8
```

### 技术栈确定
```yaml
前端技术栈:
  - React 18
  - TypeScript
  - Ant Design
  - React Query (数据获取)
  - Zustand (状态管理)

后端技术栈:
  - Next.js 14 (App Router)
  - TypeScript
  - Prisma ORM
  - NextAuth.js (认证)
  - ioredis (Redis客户端)

AI服务技术栈:
  - Python 3.11+
  - FastAPI
  - 最新嵌入模型 (待评估)
  - Milvus Python SDK
  - RabbitMQ (pika)

存储技术栈:
  - PostgreSQL 15
  - Redis 7
  - Milvus 2.3+ (现有向量数据库)
  - RabbitMQ 3.12
```

## 阶段性重构计划

### 阶段1：基础架构搭建 (1-2周)
**目标**：建立完整的开发环境和基础框架

#### 1.1 项目结构重组
```
rag-system/
├── frontend/                    # React前端
│   ├── src/
│   │   ├── components/         # UI组件
│   │   ├── pages/             # 页面组件
│   │   ├── hooks/             # 自定义Hook
│   │   ├── services/          # API服务
│   │   └── utils/             # 工具函数
│   ├── package.json
│   └── next.config.js
├── backend/                     # Next.js后端
│   ├── src/
│   │   ├── pages/api/         # API路由
│   │   ├── lib/               # 工具库
│   │   ├── services/          # 业务服务
│   │   └── middleware/        # 中间件
│   ├── prisma/                # 数据库Schema
│   └── package.json
├── python-services/             # Python AI服务
│   ├── document-processor/     # 文档处理服务
│   ├── vector-service/         # 向量化服务
│   ├── rag-service/           # RAG检索服务
│   └── shared/                # 共享代码
├── docker-compose.yml
└── README.md
```

#### 1.2 Docker环境配置
```yaml
# docker-compose.yml
version: '3.8'
services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:3001
    depends_on:
      - backend

  backend:
    build: ./backend
    ports:
      - "3001:3000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/rag_db
      - REDIS_URL=redis://redis:6379
      - RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672
    depends_on:
      - postgres
      - redis
      - rabbitmq

  document-processor:
    build: ./python-services/document-processor
    ports:
      - "8001:8001"
    depends_on:
      - rabbitmq

  vector-service:
    build: ./python-services/vector-service
    ports:
      - "8002:8002"
    environment:
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - PINECONE_ENVIRONMENT=${PINECONE_ENVIRONMENT}

  rag-service:
    build: ./python-services/rag-service
    ports:
      - "8003:8003"
    environment:
      - PINECONE_API_KEY=${PINECONE_API_KEY}

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: rag_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  rabbitmq:
    image: rabbitmq:3.12-management
    ports:
      - "5672:5672"
      - "15672:15672"
    environment:
      RABBITMQ_DEFAULT_USER: guest
      RABBITMQ_DEFAULT_PASS: guest

volumes:
  postgres_data:
```

#### 1.3 数据库设计
```sql
-- PostgreSQL Schema (prisma/schema.prisma)
model User {
  id        String   @id @default(cuid())
  email     String   @unique
  name      String?
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
  
  documents Document[]
  queries   Query[]
}

model Document {
  id            String           @id @default(cuid())
  filename      String
  originalName  String
  mimeType      String
  size          Int
  status        DocumentStatus   @default(PENDING)
  uploadedBy    String
  createdAt     DateTime         @default(now())
  updatedAt     DateTime         @updatedAt
  
  user          User            @relation(fields: [uploadedBy], references: [id])
  chunks        DocumentChunk[]
}

model DocumentChunk {
  id         String   @id @default(cuid())
  documentId String
  content    String
  chunkIndex Int
  vectorId   String   // Pinecone vector ID
  metadata   Json?
  createdAt  DateTime @default(now())
  
  document   Document @relation(fields: [documentId], references: [id])
}

model Query {
  id          String   @id @default(cuid())
  text        String
  response    String?
  userId      String
  responseTime Int?    // 响应时间(ms)
  createdAt   DateTime @default(now())
  
  user        User     @relation(fields: [userId], references: [id])
}

enum DocumentStatus {
  PENDING
  PROCESSING
  COMPLETED
  FAILED
}
```

### 阶段2：Python服务重构 (2-3周)
**目标**：将现有Python代码拆分为专业化微服务

#### 2.1 现有代码迁移策略

##### 文档处理服务 (8001端口)
```python
# python-services/document-processor/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from src.document_processor import DocumentProcessor
from src.rabbit_client import RabbitMQClient
import asyncio

app = FastAPI()
rabbit_client = RabbitMQClient()

@app.post("/process-document")
async def process_document(file: UploadFile = File(...)):
    """处理上传的文档"""
    try:
        # 使用现有的document_processor.py逻辑
        processor = DocumentProcessor()
        chunks = await processor.process_file(
            file_content=await file.read(),
            filename=file.filename,
            mime_type=file.content_type
        )
        
        # 发送处理结果到消息队列
        await rabbit_client.publish_document_processed({
            'filename': file.filename,
            'chunks': [chunk.dict() for chunk in chunks],
            'status': 'completed'
        })
        
        return {"status": "processing", "chunks_count": len(chunks)}
        
    except Exception as e:
        await rabbit_client.publish_document_processed({
            'filename': file.filename,
            'status': 'failed',
            'error': str(e)
        })
        raise HTTPException(status_code=500, detail=str(e))

# 保留现有代码：
# - src/data_processing/processors/ (完整保留)
# - src/data_processing/splitting/ (完整保留)
```

##### 向量化服务 (8002端口)
```python
# python-services/vector-service/main.py
from fastapi import FastAPI
from src.vectorizer import ModernVectorizer
from src.pinecone_client import PineconeClient
import pinecone

app = FastAPI()

class ModernVectorizer:
    """使用最新嵌入模型的向量化器"""
    
    def __init__(self):
        # 评估并使用最新模型，替换现有的BGE-M3
        # 候选模型：
        # - OpenAI text-embedding-3-large
        # - Cohere embed-v3
        # - sentence-transformers/all-MiniLM-L6-v2 (轻量级)
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 临时选择
        self.model = None
        
    def vectorize_batch(self, texts: List[str]) -> List[List[float]]:
        """批量向量化"""
        if not self.model:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

@app.post("/vectorize")
async def vectorize_texts(texts: List[str]):
    """向量化文本并存储到Milvus"""
    vectorizer = ModernVectorizer()
    milvus_client = MilvusClient()
    
    # 生成向量
    vectors = vectorizer.vectorize_batch(texts)
    
    # 存储到Milvus
    vector_ids = await milvus_client.upsert_vectors(
        vectors=vectors,
        texts=texts
    )
    
    return {"vector_ids": vector_ids}

# 迁移策略：
# - 保留 src/data_processing/vectorization/base.py 的接口设计
# - 更新向量化模型为最新版本
# - 继续使用现有的Milvus存储
```

##### RAG检索服务 (8003端口)  
```python
# python-services/rag-service/main.py
from fastapi import FastAPI
from src.rag_retriever import RAGRetriever
from src.pinecone_client import PineconeClient
from src.llm_client import LLMClient

app = FastAPI()

@app.post("/query")
async def query_documents(query: str, top_k: int = 3):
    """执行RAG查询"""
    try:
        # 向量化查询
        vectorizer = ModernVectorizer()
        query_vector = vectorizer.vectorize_batch([query])[0]
        
        # Milvus检索
        milvus_client = MilvusClient()
        similar_docs = await milvus_client.search(
            vector=query_vector,
            top_k=top_k
        )
        
        # 生成回答
        llm_client = LLMClient()  # 保留现有的ollama_client.py逻辑
        answer = await llm_client.generate_answer(
            query=query,
            documents=similar_docs
        )
        
        return {
            "answer": answer,
            "sources": similar_docs,
            "query_time": response_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 迁移策略：
# - 保留 src/rag/retriever.py 的核心逻辑
# - 保留 src/chains/processors/ 的查询处理器
# - 继续使用现有的Milvus检索
# - 保留现有的LLM调用逻辑
```

#### 2.2 Milvus客户端优化
```python
# python-services/shared/milvus_client.py
from pymilvus import Collection, connections
from typing import List, Dict, Any
import os
import numpy as np

class MilvusClient:
    def __init__(self):
        # 使用现有的Milvus配置
        connections.connect(
            alias="default",
            host=os.getenv("MILVUS_HOST", "localhost"),
            port=os.getenv("MILVUS_PORT", "19530")
        )
        
        collection_name = os.getenv("MILVUS_COLLECTION", "document_store")
        self.collection = Collection(collection_name)
        
        # 确保集合已加载
        if not self.collection.has_index():
            self._create_index()
        self.collection.load()
    
    def _create_index(self):
        """创建向量索引（如果不存在）"""
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params
        )
    
    async def upsert_vectors(self, vectors: List[List[float]], 
                           texts: List[str], 
                           metadata: List[Dict] = None) -> List[str]:
        """上传向量到Milvus"""
        import uuid
        
        vector_ids = [str(uuid.uuid4()) for _ in texts]
        
        # 准备插入数据
        entities = [
            vector_ids,  # id字段
            texts,       # text字段
            vectors,     # embedding字段
        ]
        
        # 如果有元数据，添加到实体中
        if metadata:
            # 这里需要根据你的collection schema来调整
            pass
        
        self.collection.insert(entities)
        self.collection.flush()
        
        return vector_ids
    
    async def search(self, vector: List[float], top_k: int = 3) -> List[Dict]:
        """检索相似向量"""
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        results = self.collection.search(
            data=[vector],
            anns_field="embedding", 
            param=search_params,
            limit=top_k,
            output_fields=["text", "id"]
        )
        
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "text": hit.entity.get("text"),
                    "metadata": {}  # 可以添加更多元数据
                })
        
        return formatted_results
```

### 阶段3：Next.js后端开发 (2-3周)
**目标**：构建完整的业务逻辑和API编排层

#### 3.1 API Routes设计
```typescript
// backend/src/pages/api/documents/upload.ts
import { NextApiRequest, NextApiResponse } from 'next';
import formidable from 'formidable';
import { PrismaClient } from '@prisma/client';
import { uploadToDocumentProcessor } from '@/services/pythonServices';

const prisma = new PrismaClient();

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const form = formidable({ multiples: false });
    const [fields, files] = await form.parse(req);
    const file = Array.isArray(files.file) ? files.file[0] : files.file;

    if (!file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    // 1. 保存文档记录到PostgreSQL
    const document = await prisma.document.create({
      data: {
        filename: file.originalFilename || 'unknown',
        originalName: file.originalFilename || 'unknown',
        mimeType: file.mimetype || 'application/octet-stream',
        size: file.size,
        status: 'PENDING',
        uploadedBy: req.session.user.id, // 需要认证中间件
      }
    });

    // 2. 发送到Python文档处理服务
    const processResult = await uploadToDocumentProcessor(file, document.id);

    // 3. 更新状态
    await prisma.document.update({
      where: { id: document.id },
      data: { status: 'PROCESSING' }
    });

    res.status(200).json({
      documentId: document.id,
      status: 'processing',
      message: 'Document uploaded and processing started'
    });

  } catch (error) {
    console.error('Upload error:', error);
    res.status(500).json({ error: 'Upload failed' });
  }
}

// backend/src/pages/api/query.ts
import { NextApiRequest, NextApiResponse } from 'next';
import { queryRAGService } from '@/services/pythonServices';
import { PrismaClient } from '@prisma/client';
import { redis } from '@/lib/redis';

const prisma = new PrismaClient();

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { query } = req.body;
  const userId = req.session.user.id;

  try {
    // 1. 检查缓存
    const cacheKey = `query:${Buffer.from(query).toString('base64')}`;
    const cachedResult = await redis.get(cacheKey);
    
    if (cachedResult) {
      return res.status(200).json(JSON.parse(cachedResult));
    }

    // 2. 调用Python RAG服务
    const startTime = Date.now();
    const result = await queryRAGService(query);
    const responseTime = Date.now() - startTime;

    // 3. 记录查询历史
    await prisma.query.create({
      data: {
        text: query,
        response: result.answer,
        userId,
        responseTime
      }
    });

    // 4. 缓存结果
    await redis.setex(cacheKey, 3600, JSON.stringify(result)); // 1小时缓存

    res.status(200).json(result);

  } catch (error) {
    console.error('Query error:', error);
    res.status(500).json({ error: 'Query failed' });
  }
}
```

#### 3.2 Python服务客户端
```typescript
// backend/src/services/pythonServices.ts
import axios from 'axios';

const DOCUMENT_PROCESSOR_URL = process.env.DOCUMENT_PROCESSOR_URL || 'http://localhost:8001';
const RAG_SERVICE_URL = process.env.RAG_SERVICE_URL || 'http://localhost:8003';

const VECTOR_SERVICE_URL = process.env.VECTOR_SERVICE_URL || 'http://localhost:8002';

export async function uploadToDocumentProcessor(file: any, documentId: string) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('document_id', documentId);

  const response = await axios.post(
    `${DOCUMENT_PROCESSOR_URL}/process-document`,
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 30000
    }
  );

  return response.data;
}

export async function queryRAGService(query: string) {
  const response = await axios.post(
    `${RAG_SERVICE_URL}/query`,
    { query },
    { timeout: 15000 }
  );

  return response.data;
}

export async function vectorizeTexts(texts: string[]) {
  const response = await axios.post(
    `${VECTOR_SERVICE_URL}/vectorize`,
    { texts },
    { timeout: 30000 }
  );

  return response.data;
}
```

#### 3.3 RabbitMQ消息处理
```typescript
// backend/src/services/messageQueue.ts
import amqp from 'amqplib';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export class MessageQueueService {
  private connection: amqp.Connection | null = null;
  private channel: amqp.Channel | null = null;

  async connect() {
    const rabbitmqUrl = process.env.RABBITMQ_URL || 'amqp://localhost:5672';
    this.connection = await amqp.connect(rabbitmqUrl);
    this.channel = await this.connection.createChannel();

    // 监听文档处理完成消息
    await this.channel.assertQueue('document_processed');
    await this.channel.consume('document_processed', async (msg) => {
      if (msg) {
        const data = JSON.parse(msg.content.toString());
        await this.handleDocumentProcessed(data);
        this.channel!.ack(msg);
      }
    });
  }

  private async handleDocumentProcessed(data: any) {
    const { filename, chunks, status, error } = data;

    if (status === 'completed') {
      // 更新文档状态
      await prisma.document.updateMany({
        where: { filename },
        data: { status: 'COMPLETED' }
      });

      // 保存文档块
      for (const chunk of chunks) {
        await prisma.documentChunk.create({
          data: {
            documentId: chunk.documentId,
            content: chunk.content,
            chunkIndex: chunk.index,
            vectorId: chunk.vectorId,
            metadata: chunk.metadata
          }
        });
      }
    } else if (status === 'failed') {
      await prisma.document.updateMany({
        where: { filename },
        data: { status: 'FAILED' }
      });
    }
  }
}

// 启动消息队列服务
const messageQueue = new MessageQueueService();
messageQueue.connect().catch(console.error);
```

### 阶段4：React前端开发 (2-3周)
**目标**：构建现代化用户界面

#### 4.1 页面结构设计
```typescript
// frontend/src/components/DocumentUpload.tsx
import React, { useState } from 'react';
import { Upload, Button, message, Progress } from 'antd';
import { InboxOutlined } from '@ant-design/icons';
import { useDocumentUpload } from '@/hooks/useDocuments';

const { Dragger } = Upload;

export const DocumentUpload: React.FC = () => {
  const [uploadProgress, setUploadProgress] = useState(0);
  const { uploadDocument, isUploading } = useDocumentUpload();

  const handleUpload = async (file: File) => {
    try {
      setUploadProgress(0);
      
      const result = await uploadDocument(file, (progress) => {
        setUploadProgress(progress);
      });
      
      message.success(`文档 ${file.name} 上传成功！`);
      setUploadProgress(0);
      
    } catch (error) {
      message.error('上传失败，请重试');
      setUploadProgress(0);
    }
  };

  return (
    <div className="upload-container">
      <Dragger
        name="file"
        multiple={false}
        accept=".pdf,.docx,.txt,.md"
        beforeUpload={(file) => {
          handleUpload(file);
          return false; // 阻止默认上传
        }}
        disabled={isUploading}
      >
        <p className="ant-upload-drag-icon">
          <InboxOutlined />
        </p>
        <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
        <p className="ant-upload-hint">
          支持 PDF, Word, TXT, Markdown 格式
        </p>
      </Dragger>
      
      {uploadProgress > 0 && (
        <Progress 
          percent={uploadProgress} 
          status={uploadProgress === 100 ? "success" : "active"}
          style={{ marginTop: 16 }}
        />
      )}
    </div>
  );
};

// frontend/src/components/ChatInterface.tsx
import React, { useState } from 'react';
import { Input, Button, Card, Spin, Typography } from 'antd';
import { SendOutlined } from '@ant-design/icons';
import { useQuery } from '@/hooks/useQuery';

const { TextArea } = Input;
const { Paragraph } = Typography;

export const ChatInterface: React.FC = () => {
  const [query, setQuery] = useState('');
  const [chatHistory, setChatHistory] = useState<any[]>([]);
  const { executeQuery, isQuerying } = useQuery();

  const handleQuery = async () => {
    if (!query.trim()) return;

    const userMessage = { type: 'user', content: query, timestamp: Date.now() };
    setChatHistory(prev => [...prev, userMessage]);

    try {
      const result = await executeQuery(query);
      const botMessage = {
        type: 'bot',
        content: result.answer,
        sources: result.sources,
        timestamp: Date.now()
      };
      setChatHistory(prev => [...prev, botMessage]);
      
    } catch (error) {
      const errorMessage = {
        type: 'error',
        content: '查询失败，请重试',
        timestamp: Date.now()
      };
      setChatHistory(prev => [...prev, errorMessage]);
    }

    setQuery('');
  };

  return (
    <div className="chat-container">
      <div className="chat-history">
        {chatHistory.map((message, index) => (
          <Card 
            key={index}
            className={`message ${message.type}`}
            style={{ marginBottom: 16 }}
          >
            <Paragraph>{message.content}</Paragraph>
            {message.sources && (
              <div className="sources">
                <strong>参考来源：</strong>
                {message.sources.map((source: any, idx: number) => (
                  <div key={idx} className="source-item">
                    {source.text.substring(0, 100)}...
                  </div>
                ))}
              </div>
            )}
          </Card>
        ))}
        {isQuerying && <Spin size="large" />}
      </div>
      
      <div className="query-input">
        <TextArea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="输入你的问题..."
          autoSize={{ minRows: 2, maxRows: 4 }}
          onPressEnter={(e) => {
            if (!e.shiftKey) {
              e.preventDefault();
              handleQuery();
            }
          }}
        />
        <Button
          type="primary"
          icon={<SendOutlined />}
          onClick={handleQuery}
          loading={isQuerying}
          disabled={!query.trim()}
        >
          发送
        </Button>
      </div>
    </div>
  );
};
```

#### 4.2 自定义Hooks
```typescript
// frontend/src/hooks/useDocuments.ts
import { useState } from 'react';
import { message } from 'antd';

export const useDocumentUpload = () => {
  const [isUploading, setIsUploading] = useState(false);

  const uploadDocument = async (
    file: File, 
    onProgress?: (progress: number) => void
  ) => {
    setIsUploading(true);
    
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/documents/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const result = await response.json();
      onProgress?.(100);
      
      return result;
      
    } finally {
      setIsUploading(false);
    }
  };

  return { uploadDocument, isUploading };
};

// frontend/src/hooks/useQuery.ts
import { useState } from 'react';
import { useQuery as useReactQuery } from 'react-query';

export const useQuery = () => {
  const [isQuerying, setIsQuerying] = useState(false);

  const executeQuery = async (query: string) => {
    setIsQuerying(true);
    
    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        throw new Error('Query failed');
      }

      return await response.json();
      
    } finally {
      setIsQuerying(false);
    }
  };

  return { executeQuery, isQuerying };
};
```

### 阶段5：集成测试与优化 (1-2周)
**目标**：确保整个系统稳定运行并进行性能优化

#### 5.1 端到端测试流程
```typescript
// 测试流程脚本
const e2eTestFlow = {
  "1. 文档上传测试": async () => {
    // 上传PDF文档
    // 验证文档状态变化：PENDING → PROCESSING → COMPLETED
    // 验证PostgreSQL中的记录
    // 验证Pinecone中的向量存储
  },
  
  "2. 查询功能测试": async () => {
    // 发送查询请求
    // 验证返回结果格式
    // 验证响应时间 < 3秒
    // 验证缓存机制生效
  },
  
  "3. 并发性能测试": async () => {
    // 模拟10个并发上传
    // 模拟50个并发查询
    // 验证系统稳定性
  }
};
```

#### 5.2 性能监控配置
```typescript
// backend/src/middleware/monitoring.ts
import { NextApiRequest, NextApiResponse } from 'next';
import { performance } from 'perf_hooks';

export const monitoringMiddleware = (handler: any) => {
  return async (req: NextApiRequest, res: NextApiResponse) => {
    const startTime = performance.now();
    
    // 记录请求
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
    
    try {
      await handler(req, res);
      
      const duration = performance.now() - startTime;
      console.log(`Request completed in ${duration.toFixed(2)}ms`);
      
      // TODO: 发送到监控系统（预留接口）
      
    } catch (error) {
      const duration = performance.now() - startTime;
      console.error(`Request failed in ${duration.toFixed(2)}ms:`, error);
      throw error;
    }
  };
};
```

## 现有代码迁移映射

### 保留的核心代码
```python
# 完全保留 (移动到对应服务)
src/data_processing/processors/          → document-processor/src/processors/
src/data_processing/splitting/           → document-processor/src/splitting/
src/chains/processors/                   → rag-service/src/processors/
src/model/ollama_client.py              → rag-service/src/llm_client.py
src/utils/logging_utils.py              → shared/utils/

# 重构保留 (接口保持兼容)
src/data_processing/vectorization/      → vector-service/src/ (更新模型)
src/rag/retriever.py                    → rag-service/src/ (继续使用Milvus)
src/data_processing/storage/milvus_store.py → shared/milvus_client.py (优化)

# 废弃代码
src/api/                                # 替换为Next.js API
src/data_processing/storage/faiss_store.py  # 专注使用Milvus
config/config.py                        # 分散到各服务的配置中
```

### 向量化模型升级建议
```python
# 当前模型评估和替换
current_models = {
    "BGE-M3": "综合性能好，但模型较大",
    "BERT": "经典模型，性能一般", 
    "TF-IDF": "传统方法，性能有限"
}

recommended_models = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "优势": "轻量级，速度快，性能不错",
        "维度": 384,
        "适用": "快速原型和资源受限环境"
    },
    "text-embedding-3-small": {
        "优势": "OpenAI最新模型，性能优秀",
        "维度": 1536,
        "适用": "追求最佳效果，有API成本预算"
    },
    "BAAI/bge-large-en-v1.5": {
        "优势": "开源模型中性能最佳之一",
        "维度": 1024,
        "适用": "平衡性能和成本"
    }
}

# 建议：先使用all-MiniLM-L6-v2快速验证，后续可替换为更强模型
```

## 部署和运维

### 环境变量配置
```bash
# .env.example
# Database
DATABASE_URL="postgresql://postgres:password@localhost:5432/rag_db"
REDIS_URL="redis://localhost:6379"
RABBITMQ_URL="amqp://guest:guest@localhost:5672"

# Milvus (现有配置)
MILVUS_HOST="localhost"
MILVUS_PORT="19530"
MILVUS_COLLECTION="document_store"

# Python Services
DOCUMENT_PROCESSOR_URL="http://localhost:8001"
VECTOR_SERVICE_URL="http://localhost:8002" 
RAG_SERVICE_URL="http://localhost:8003"

# NextAuth
NEXTAUTH_URL="http://localhost:3001"
NEXTAUTH_SECRET="your-nextauth-secret"

# Embedding Model
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

### 快速启动脚本
```bash
#!/bin/bash
# scripts/start-dev.sh

echo "🚀 启动RAG系统开发环境..."

# 1. 启动基础服务（包含现有的Milvus）
echo "启动数据库、Milvus和消息队列..."
docker-compose up -d postgres redis rabbitmq milvus etcd minio

# 2. 启动Python服务
echo "启动Python AI服务..."
docker-compose up -d document-processor vector-service rag-service

# 3. 启动Next.js后端
echo "启动Next.js后端..."
cd backend && npm run dev &

# 4. 启动React前端
echo "启动React前端..."
cd frontend && npm run dev &

echo "✅ 系统启动完成！"
echo "前端地址: http://localhost:3000"
echo "后端API: http://localhost:3001"
echo "文档处理服务: http://localhost:8001"
echo "向量化服务: http://localhost:8002"  
echo "RAG服务: http://localhost:8003"
```

## 成功验收标准

### 阶段1验收
- ✅ Docker环境正常启动
- ✅ 所有服务健康检查通过
- ✅ 数据库连接正常
- ✅ 基础API响应正常

### 阶段2验收  
- ✅ 文档上传处理正常
- ✅ 向量化服务正常工作
- ✅ Milvus存储和检索正常
- ✅ RAG查询返回正确结果

### 阶段3验收
- ✅ Next.js API全部正常
- ✅ 数据库CRUD操作正常
- ✅ 缓存机制正常工作
- ✅ 消息队列处理正常

### 阶段4验收
- ✅ 前端页面正常渲染
- ✅ 文档上传界面完整
- ✅ 聊天界面功能正常
- ✅ 用户体验流畅

### 最终验收
- ✅ 完整的端到端流程正常
- ✅ 并发处理能力满足要求
- ✅ 错误处理机制完善
- ✅ 系统稳定性达标

---

## 开发协作建议

由于你是独立开发者，建议按以下顺序执行：

1. **优先搭建基础架构**：先把Docker环境和数据库弄好
2. **重构Python服务**：利用现有代码快速构建微服务
3. **开发Next.js后端**：重点关注API设计和业务逻辑
4. **构建React前端**：最后完善用户界面

每个阶段完成后都应该有可以演示的功能，这样可以快速验证架构设计的正确性，及时调整方向。

有任何问题随时讨论，我会协助你完成整个重构过程！

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "\u786e\u8ba4\u91cd\u6784\u65b9\u6848\u7684\u5173\u952e\u7ec6\u8282", "id": "1", "priority": "high", "status": "completed"}, {"content": "\u5206\u6790\u5f53\u524d\u4ee3\u7801\u7684\u8fc1\u79fb\u7b56\u7565", "id": "2", "priority": "high", "status": "completed"}, {"content": "\u8bbe\u8ba1\u8be6\u7ec6\u7684\u91cd\u6784\u6307\u5bfc\u6587\u6863", "id": "3", "priority": "high", "status": "completed"}]