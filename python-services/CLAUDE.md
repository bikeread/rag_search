# Python服务集群 - 开发指导文档

## 📅 最新更新 (2025-08-13)

### ✅ 系统集成测试完成
- **E2E验证**: 全链路功能测试通过
- **性能表现**: 
  - 文档处理: <2秒/文档
  - 向量检索: <500ms
  - RAG查询: ~5秒（含LLM生成）
- **稳定性**: 服务运行稳定，无内存泄漏

### 🔍 已知优化点
- **查询精度**: 需要优化语义理解（Phase 2计划中）
- **响应速度**: 考虑引入缓存机制
- **向量维度**: 当前384维，可能需要调整

## 📋 服务集群概览

**Python Services Cluster** 是RAG系统的核心AI处理引擎，由三个专业化的FastAPI微服务组成，负责文档处理、向量化存储和智能问答的完整流程。采用异步架构和现代Python技术栈，提供高性能的AI服务支持。

### 🏗️ 服务架构
```
┌─────────────────────────────────────────────────────────────────┐
│                  Python Services Ecosystem                      │
├─────────────────┬─────────────────┬─────────────────────────────┤
│ Document        │ Vector          │ RAG                         │
│ Processor       │ Service         │ Service                     │
│ :8001          │ :8002          │ :8003                       │
│                 │                 │                             │
│ ┌─────────────┐ │ ┌─────────────┐ │ ┌─────────────────────────┐ │
│ │File Upload  │ │ │TF-IDF       │ │ │Query Processing         │ │
│ │Text Extract │─┼→│Vectorizer   │─┼→│Answer Generation        │ │
│ │Chunking     │ │ │Milvus Store │ │ │Context Building         │ │
│ └─────────────┘ │ └─────────────┘ │ └─────────────────────────┘ │
└─────────────────┴─────────────────┴─────────────────────────────┘
         │                   │                       │
         ▼                   ▼                       ▼
┌─────────────────┬─────────────────┬─────────────────────────────┐
│   RabbitMQ      │    Milvus       │       Ollama               │
│ :5672          │ :19530         │      :11434                │
└─────────────────┴─────────────────┴─────────────────────────────┘
```

### 🎯 技术栈统一
- **框架**: FastAPI + Uvicorn + Pydantic
- **异步**: asyncio + aiohttp + httpx
- **AI/ML**: scikit-learn + numpy + TF-IDF
- **向量数据库**: Milvus + pymilvus
- **LLM集成**: Ollama (llama3.2:1b)
- **消息队列**: RabbitMQ + aio-pika
- **日志**: 结构化日志 + JSON格式
- **配置**: 环境变量 + dataclass配置

## 🔧 核心服务详解

### 📄 Document Processor (:8001)
**职责**: 文档解析、文本分块、异步任务协调

**核心功能**:
- 多格式文档解析 (PDF, DOCX, TXT, MD)
- 智能文本分块 (RecursiveTextSplitter)
- Vector Service异步调用
- RabbitMQ消息发布

**开发要点**:
```python
# 核心API端点
@app.post("/process-document")
async def process_document(
    file: UploadFile,
    document_id: str = None,
    enable_chunking: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
)

# 处理流程
1. 文件验证和解析
2. 文本提取和清理
3. 智能分块处理
4. 调用向量化服务
5. 发布处理状态
```

### 🔢 Vector Service (:8002)  
**职责**: 文本向量化、向量存储、相似性搜索

**核心功能**:
- CPU友好TF-IDF向量化 (384维)
- Milvus向量数据库集成
- 批量向量处理
- 高性能向量搜索

**开发要点**:
```python
# 核心API端点
@app.post("/vectorize")
async def vectorize_texts(request: VectorizeRequest)

@app.post("/search")
async def search_vectors(request: SearchRequest)

# TF-IDF向量化器
class CPUVectorizer:
    def __init__(self, vector_dim=384):
        self.vectorizer = TfidfVectorizer(
            max_features=vector_dim,
            ngram_range=(1, 2),
            stop_words='english'
        )
```

### 🤖 RAG Service (:8003)
**职责**: 查询处理、检索协调、答案生成

**核心功能**:
- RAG完整流程 (检索→上下文→生成)
- Ollama LLM集成
- 智能上下文构建
- 多语言查询支持

**开发要点**:
```python
# 核心API端点
@app.post("/query")
async def rag_query(request: QueryRequest)

# RAG处理流程
async def process_rag_query(query: str, top_k: int = 3):
    # 1. 向量检索
    search_results = await vector_client.search(query, top_k)
    
    # 2. 上下文构建
    context = build_context(search_results)
    
    # 3. LLM生成
    answer = await ollama_client.generate(context, query)
    
    return RAGResponse(answer=answer, sources=search_results)
```

## 🧪 开发和调试指南

### 环境配置
```bash
# 共享环境变量
PYTHONPATH=/app/src:/app/shared
LOG_LEVEL=INFO
ASYNC_TIMEOUT=30

# Document Processor环境变量
HOST=0.0.0.0
PORT=8001
RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672
VECTOR_SERVICE_URL=http://vector-service:8002

# Vector Service环境变量  
HOST=0.0.0.0
PORT=8002
MILVUS_HOST=milvus
MILVUS_PORT=19530
MILVUS_COLLECTION=rag_documents
VECTOR_DIM=384

# RAG Service环境变量
HOST=0.0.0.0
PORT=8003  
VECTOR_SERVICE_URL=http://vector-service:8002
OLLAMA_URL=http://ollama:11434
OLLAMA_MODEL=llama3.2:1b
```

### 本地开发启动
```bash
# 1. 启动依赖服务
docker-compose up -d postgres redis rabbitmq milvus ollama

# 2. 安装Python依赖
cd python-services
pip install -r shared/requirements.txt

# 3. 启动各个服务
cd document-processor && python main.py  # 终端1
cd vector-service && python main.py      # 终端2  
cd rag-service && python main.py         # 终端3

# 4. 健康检查
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
```

### Docker开发模式
```bash
# 启动全部Python服务
docker-compose up -d document-processor vector-service rag-service

# 查看服务日志
docker-compose logs -f document-processor
docker-compose logs -f vector-service
docker-compose logs -f rag-service

# 重启特定服务
docker-compose restart vector-service
```

## 🔄 服务间通信协议

### 1. 文档处理流程
```mermaid
sequenceDiagram
    participant Backend as Next.js Backend
    participant DocProcessor as Document Processor
    participant VectorService as Vector Service
    participant Milvus as Milvus DB
    
    Backend->>DocProcessor: POST /process-document
    DocProcessor->>DocProcessor: Parse & Chunk Document
    DocProcessor->>VectorService: POST /vectorize
    VectorService->>Milvus: Store Vectors
    VectorService->>DocProcessor: Return Success
    DocProcessor->>Backend: Return Document Status
```

### 2. RAG查询流程
```mermaid
sequenceDiagram
    participant Backend as Next.js Backend
    participant RAGService as RAG Service
    participant VectorService as Vector Service
    participant Ollama as Ollama LLM
    
    Backend->>RAGService: POST /query
    RAGService->>VectorService: POST /search
    VectorService->>RAGService: Return Similar Documents
    RAGService->>Ollama: Generate Answer
    Ollama->>RAGService: Return Generated Text
    RAGService->>Backend: Return Complete Response
```

## 🧩 共享组件和工具

### 1. 配置管理
```python
# shared/config.py - 统一配置管理
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class ServiceConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    reload: bool = False
    
    @classmethod
    def from_env(cls, port_default: int = 8000):
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", port_default)),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            reload=os.getenv("RELOAD", "false").lower() == "true"
        )

@dataclass  
class MilvusConfig:
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "rag_documents"
    vector_dim: int = 384
    
    @classmethod
    def from_env(cls):
        return cls(
            host=os.getenv("MILVUS_HOST", "localhost"),
            port=int(os.getenv("MILVUS_PORT", "19530")),
            collection_name=os.getenv("MILVUS_COLLECTION", "rag_documents"),
            vector_dim=int(os.getenv("VECTOR_DIM", "384"))
        )
```

### 2. Milvus客户端
```python
# shared/milvus_client.py - 统一向量数据库客户端
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from typing import List, Dict, Any
import logging

class MilvusClient:
    def __init__(self, config: MilvusConfig):
        self.config = config
        self.collection = None
        
    async def connect(self):
        try:
            connections.connect(
                alias="default",
                host=self.config.host,
                port=self.config.port
            )
            await self._setup_collection()
            logging.info("Milvus连接成功")
        except Exception as e:
            logging.error(f"Milvus连接失败: {e}")
            
    async def _setup_collection(self):
        # 定义Collection Schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.config.vector_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        
        schema = CollectionSchema(fields, description="RAG文档向量存储")
        
        # 创建或加载Collection
        if not Collection.exists(self.config.collection_name):
            collection = Collection(self.config.collection_name, schema)
            
            # 创建向量索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index("vector", index_params)
            
        self.collection = Collection(self.config.collection_name)
        self.collection.load()
        
    async def insert_vectors(self, data: List[Dict[str, Any]]):
        if not self.collection:
            raise RuntimeError("Milvus未连接")
            
        entities = [
            [item["id"] for item in data],
            [item["vector"] for item in data], 
            [item["text"] for item in data],
            [item["document_id"] for item in data],
            [item.get("metadata", {}) for item in data]
        ]
        
        result = self.collection.insert(entities)
        self.collection.flush()
        return result
        
    async def search_vectors(self, query_vector: List[float], top_k: int = 5):
        if not self.collection:
            raise RuntimeError("Milvus未连接")
            
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        results = self.collection.search(
            data=[query_vector],
            anns_field="vector", 
            param=search_params,
            limit=top_k,
            output_fields=["text", "document_id", "metadata"]
        )
        
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "text": hit.entity.get("text"),
                "document_id": hit.entity.get("document_id"), 
                "metadata": hit.entity.get("metadata", {})
            }
            for hit in results[0]
        ]
```

### 3. RabbitMQ客户端
```python
# shared/rabbit_client.py - 消息队列客户端
import aio_pika
import json
from typing import Dict, Any
import logging

class RabbitMQClient:
    def __init__(self, rabbitmq_url: str):
        self.rabbitmq_url = rabbitmq_url
        self.connection = None
        self.channel = None
        
    async def connect(self):
        try:
            self.connection = await aio_pika.connect_robust(self.rabbitmq_url)
            self.channel = await self.connection.channel()
            logging.info("RabbitMQ连接成功")
        except Exception as e:
            logging.error(f"RabbitMQ连接失败: {e}")
            
    async def publish_message(self, queue_name: str, message: Dict[str, Any]):
        if not self.channel:
            await self.connect()
            
        queue = await self.channel.declare_queue(queue_name, durable=True)
        
        await self.channel.default_exchange.publish(
            aio_pika.Message(
                json.dumps(message, ensure_ascii=False).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT
            ),
            routing_key=queue_name
        )
        
    async def close(self):
        if self.connection:
            await self.connection.close()
```

### 4. 统一日志配置
```python
# shared/logging_config.py - 结构化日志
import logging
import json
from datetime import datetime
from typing import Any, Dict

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "service": getattr(record, "service", "unknown"),
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)
            
        return json.dumps(log_entry, ensure_ascii=False)

def setup_logging(service_name: str, log_level: str = "INFO"):
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))
    
    # 清除现有handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 添加JSON格式handler
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    # 设置服务名称
    logger = logging.LoggerAdapter(logger, {"service": service_name})
    return logger
```

## 🧪 测试和调试

### 健康检查脚本
```bash
#!/bin/bash
# scripts/health_check.sh - 服务健康检查

services=(
    "document-processor:8001"
    "vector-service:8002"
    "rag-service:8003"
)

echo "Python服务健康检查 $(date)"
echo "=============================="

for service in "${services[@]}"; do
    name=${service%%:*}
    port=${service##*:}
    
    echo -n "检查 $name ... "
    
    if response=$(curl -s --max-time 5 "http://localhost:$port/health" 2>/dev/null); then
        status=$(echo "$response" | jq -r '.status' 2>/dev/null)
        if [ "$status" = "healthy" ]; then
            echo "✅ 健康"
        else
            echo "❌ 异常: $status"
        fi
    else
        echo "❌ 连接失败"
    fi
done
```

### 端到端测试
```python
# tests/integration/test_e2e_flow.py
import pytest
import httpx
import asyncio

class TestE2EFlow:
    @pytest.mark.asyncio
    async def test_complete_rag_pipeline(self):
        """测试完整的RAG流程"""
        
        # 1. 上传文档到Document Processor
        with open("test_document.txt", "rb") as f:
            files = {"file": f}
            data = {"document_id": "e2e-test-doc"}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8001/process-document",
                    files=files,
                    data=data
                )
                assert response.status_code == 200
                
        # 2. 等待文档处理完成
        await asyncio.sleep(10)
        
        # 3. 测试向量搜索
        async with httpx.AsyncClient() as client:
            search_response = await client.post(
                "http://localhost:8002/search",
                json={
                    "query_text": "测试文档内容",
                    "top_k": 3
                }
            )
            assert search_response.status_code == 200
            search_data = search_response.json()
            assert len(search_data["results"]) > 0
            
        # 4. 测试RAG查询
        async with httpx.AsyncClient() as client:
            rag_response = await client.post(
                "http://localhost:8003/query",
                json={
                    "query": "请总结文档的主要内容",
                    "top_k": 3
                }
            )
            assert rag_response.status_code == 200
            rag_data = rag_response.json()
            assert rag_data["status"] == "completed"
            assert len(rag_data["answer"]) > 10
```

### 性能监控
```python
# shared/utils/monitoring.py - 性能监控工具
import time
import psutil
from functools import wraps
from typing import Callable
import logging

def monitor_performance(func: Callable):
    """性能监控装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # 记录开始时间和资源使用
        start_time = time.time()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = process.cpu_percent()
        
        try:
            result = await func(*args, **kwargs)
            
            # 记录性能指标
            end_time = time.time()
            duration = end_time - start_time
            end_memory = process.memory_info().rss / 1024 / 1024
            memory_delta = end_memory - start_memory
            
            logging.info(
                f"性能监控",
                extra={
                    "function": func.__name__,
                    "duration_ms": round(duration * 1000, 2),
                    "memory_usage_mb": round(end_memory, 2),
                    "memory_delta_mb": round(memory_delta, 2),
                    "cpu_percent": process.cpu_percent()
                }
            )
            
            return result
            
        except Exception as e:
            logging.error(f"函数执行异常: {func.__name__}", extra={
                "error": str(e),
                "duration_ms": round((time.time() - start_time) * 1000, 2)
            })
            raise
            
    return wrapper
```

## 📋 开发检查清单

### ✅ 服务开发规范
- [ ] 使用FastAPI + Pydantic数据验证
- [ ] 实现异步处理和错误处理
- [ ] 添加结构化JSON日志
- [ ] 实现健康检查端点
- [ ] 添加性能监控装饰器
- [ ] 遵循RESTful API设计

### ✅ AI/ML集成
- [ ] 使用CPU友好的向量化器
- [ ] 实现批量处理优化
- [ ] 添加向量维度验证
- [ ] 实现相似性搜索缓存
- [ ] 优化LLM提示工程

### ✅ 数据库集成
- [ ] 实现Milvus连接池
- [ ] 添加向量索引优化
- [ ] 实现批量向量插入
- [ ] 添加数据一致性检查
- [ ] 实现故障恢复机制

### ✅ 微服务协作
- [ ] 实现服务间HTTP通信
- [ ] 添加RabbitMQ异步消息
- [ ] 实现服务降级策略
- [ ] 添加重试机制
- [ ] 实现熔断器模式

## 🚀 部署和运维

### Docker化部署
```dockerfile
# 通用Python服务Dockerfile模板
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
COPY shared/requirements.txt shared/
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r shared/requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8001

# 启动命令
CMD ["python", "main.py"]
```

### 生产环境配置
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  document-processor:
    build: ./python-services/document-processor
    environment:
      - LOG_LEVEL=WARNING
      - WORKERS=4
      - MAX_REQUESTS=1000
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 1GB
          cpus: '1.0'
    restart: unless-stopped
    
  vector-service:
    build: ./python-services/vector-service  
    environment:
      - LOG_LEVEL=WARNING
      - VECTOR_CACHE_SIZE=10000
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 2GB
          cpus: '1.5'
    restart: unless-stopped
```

## 🔗 相关文档

### 📚 详细服务文档
- [Document Processor开发指导](document-processor/CLAUDE.md)
- [Vector Service开发指导](vector-service/CLAUDE.md)
- [RAG Service开发指导](rag-service/CLAUDE.md)

### 🛠️ 集成文档
- [后端API集成](../backend/CLAUDE.md)
- [前端界面集成](../frontend/CLAUDE.md)
- [Docker部署配置](../docker-compose.yml)

### 📈 运维文档
- [监控配置](../config/grafana/)
- [日志分析](../config/prometheus.yml)
- [健康检查脚本](./scripts/health_check.sh)

---

## 🔗 相关文档和优化方案

### 📚 详细服务文档
- [Document Processor开发指导](document-processor/CLAUDE.md)
- [Vector Service开发指导](vector-service/CLAUDE.md)
- [RAG Service开发指导](rag-service/CLAUDE.md)

### 🛠️ 集成文档
- [后端API集成](../backend/CLAUDE.md)
- [前端界面集成](../frontend/CLAUDE.md)
- [Docker部署配置](../docker-compose.yml)

### 📈 运维文档
- [监控配置](../config/grafana/)
- [日志分析](../config/prometheus.yml)
- [健康检查脚本](./scripts/health_check.sh)

### 🚀 **RAG系统准确率优化方案**
**重要**: 本Python服务集群正在实施[RAG系统准确率优化方案](../CLAUDE.md#rag系统准确率优化方案)，目标将准确率从53.4%-65%提升至90-95%。

#### 当前实施状态
- ✅ **Phase 1基础优化**: 混合检索系统、语义分块、查询改写
- 🔄 **Phase 2深度优化**: Cross-encoder重排序、嵌入模型微调 (进行中)
- 📋 **Phase 3高级增强**: GraphRAG知识图谱集成 (计划中)

#### 服务级优化重点
- **RAG Service (8003)**: 混合检索、重排序、查询优化
- **Vector Service (8002)**: 嵌入模型微调、索引优化
- **Document Processor (8001)**: 语义分块、元数据增强

详细技术方案和实施路线图请参考[根目录CLAUDE.md](../CLAUDE.md)。

---

**维护说明**: 本文档是Python服务集群的开发总指导，侧重于架构设计和开发规范。各个服务的详细开发指导和API规范请参考对应目录下的CLAUDE.md文件。所有服务修改都应该在此文档中同步更新架构信息。