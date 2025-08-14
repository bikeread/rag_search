# Document Processor Service - 开发指导文档

## 📋 服务概览

**Document Processor Service** 是RAG系统的文档处理微服务，负责文档上传、解析、分块和向量化协调。采用异步架构设计，支持多种文档格式处理，并与Vector Service集成实现端到端的文档向量化流程。

### 🏗️ 技术栈
- **框架**: FastAPI + Uvicorn + Pydantic
- **异步处理**: asyncio + BackgroundTasks
- **文档处理**: Unstructured + 自定义处理器
- **消息队列**: RabbitMQ (aio-pika)
- **HTTP客户端**: aiohttp
- **配置管理**: Pydantic Settings
- **端口**: 8001

### 📁 项目结构
```
document-processor/
├── main.py                    # FastAPI应用和路由
├── config.py                  # Pydantic配置管理
├── rabbit_client.py           # RabbitMQ消息客户端
├── logging_config.py          # 日志配置
├── milvus_client.py          # Milvus数据库客户端
├── src/
│   ├── document_processor.py  # 核心文档处理器
│   ├── processors/            # 文档格式处理器
│   │   ├── base.py           # 基础处理器接口
│   │   ├── unstructured_processor.py  # Unstructured集成
│   │   └── table_processor.py # 表格处理器
│   └── splitting/             # 文本分块策略
│       ├── splitter.py       # 分块接口
│       └── strategies.py     # 分块算法实现
├── utils/                     # 工具库
│   ├── exceptions.py         # 异常定义
│   ├── validators.py         # 输入验证
│   └── logging_utils.py      # 日志工具
└── requirements.txt          # Python依赖
```

## 🎯 核心功能模块

### 📄 1. 文档处理 (`POST /process-document`)
- **多格式支持**: PDF, DOCX, TXT, MD文档解析
- **智能分块**: RecursiveTextSplitter文本分块
- **异步处理**: BackgroundTasks后台处理
- **向量化集成**: 自动调用Vector Service

### 🔄 2. 异步任务协调
- **RabbitMQ集成**: 处理状态消息发布
- **Vector Service调用**: 文本向量化和存储
- **状态管理**: 处理进度跟踪和状态更新

### 📊 3. 监控和状态
- **健康检查**: `/health` - 服务和依赖状态
- **处理状态**: `/processing-status/{id}` - 文档处理进度
- **重新处理**: `/reprocess-document/{id}` - 文档重新处理

## 🧪 开发和调试指南

### 环境配置
```bash
# 核心服务配置
HOST=0.0.0.0
PORT=8001
RELOAD=true
DEBUG_MODE=true

# RabbitMQ配置
RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672
RABBITMQ_EXCHANGE=rag_exchange
RABBITMQ_DOCUMENT_QUEUE=document_processed

# Vector Service集成
VECTOR_SERVICE_URL=http://vector-service:8002
DOCUMENT_PROCESSOR_URL=http://document-processor:8001

# 处理配置
MAX_DOCUMENT_SIZE=52428800  # 50MB
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT=30

# 日志配置
LOG_LEVEL=INFO
ENABLE_JSON_LOGS=true
```

### 本地开发启动
```bash
# 1. 安装依赖
pip install -r requirements.txt
pip install -r ../shared/requirements.txt

# 2. 启动依赖服务
docker-compose up -d rabbitmq vector-service milvus

# 3. 启动开发服务器
python main.py

# 4. 验证服务状态
curl http://localhost:8001/health

# 服务地址: http://localhost:8001
# API文档: http://localhost:8001/docs
```

### Docker开发模式
```bash
# 构建并启动服务
docker-compose up -d document-processor

# 查看服务日志
docker-compose logs -f document-processor

# 重启服务
docker-compose restart document-processor

# 进入容器调试
docker-compose exec document-processor bash
```

## 🔧 核心组件详解

### 1. 文档处理器架构
```python
# src/document_processor.py - 统一文档处理接口
class ModernDocumentProcessor:
    def __init__(self):
        self.processors = {
            'application/pdf': PDFProcessor(),
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DOCXProcessor(),
            'text/plain': TextProcessor(),
            'text/markdown': MarkdownProcessor()
        }
        self.splitter = RecursiveTextSplitter()
    
    async def process_file(
        self,
        file_content: bytes,
        filename: str,
        mime_type: str,
        document_id: str,
        enable_chunking: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> ProcessingResult:
        """处理文档文件的核心方法"""
        # 1. 选择合适的处理器
        processor = self.processors.get(mime_type)
        if not processor:
            raise UnsupportedFormatError(f"不支持的文档格式: {mime_type}")
        
        # 2. 提取文本内容
        text_content = await processor.extract_text(file_content)
        
        # 3. 文本分块处理
        chunks = []
        if enable_chunking and text_content.strip():
            chunks = self.splitter.split_text(
                text=text_content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        
        # 4. 构建结果对象
        return ProcessingResult(
            document_id=document_id,
            filename=filename,
            status="completed",
            chunks=[
                DocumentChunk(
                    chunk_id=f"{document_id}-chunk-{i}",
                    content=chunk,
                    chunk_index=i,
                    metadata={"source": filename}
                ) for i, chunk in enumerate(chunks)
            ],
            chunk_count=len(chunks)
        )
```

### 2. Vector Service客户端
```python
# Vector Service集成客户端
class VectorServiceClient:
    def __init__(self, base_url: str = "http://vector-service:8002"):
        self.base_url = base_url
        self.session = None
    
    async def vectorize_and_store(
        self, 
        texts: List[str], 
        document_id: str
    ) -> Dict[str, Any]:
        """向量化文本并存储到Milvus"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "texts": texts,
                    "store_vectors": True,
                    "document_ids": [document_id] * len(texts),
                    "metadata": [
                        {"document_id": document_id, "chunk_index": i}
                        for i in range(len(texts))
                    ]
                }
                
                async with session.post(
                    f"{self.base_url}/vectorize",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"成功向量化 {len(texts)} 个文本块")
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"向量化失败: {response.status} - {error_text}")
                        return {"status": "failed", "error": error_text}
                        
        except asyncio.TimeoutError:
            logger.error("向量化服务调用超时")
            return {"status": "timeout", "error": "服务调用超时"}
        except Exception as e:
            logger.error(f"向量化服务调用异常: {str(e)}")
            return {"status": "failed", "error": str(e)}
```

### 3. RabbitMQ消息发布
```python
# rabbit_client.py - 消息队列客户端
class RabbitMQClient:
    def __init__(self, rabbitmq_url: str = None):
        self.rabbitmq_url = rabbitmq_url or settings.rabbitmq_url
        self.connection = None
        self.channel = None
        self.is_connected = False
    
    async def connect(self):
        """连接到RabbitMQ"""
        try:
            self.connection = await aio_pika.connect_robust(self.rabbitmq_url)
            self.channel = await self.connection.channel()
            
            # 声明交换机和队列
            await self.channel.declare_exchange(
                settings.rabbitmq_exchange,
                aio_pika.ExchangeType.TOPIC,
                durable=True
            )
            
            self.is_connected = True
            logger.info("RabbitMQ连接成功")
            
        except Exception as e:
            logger.error(f"RabbitMQ连接失败: {str(e)}")
            self.is_connected = False
            raise
    
    async def publish_document_processed(self, message: Dict[str, Any]):
        """发布文档处理完成消息"""
        if not self.is_connected:
            await self.connect()
        
        try:
            exchange = await self.channel.get_exchange(settings.rabbitmq_exchange)
            
            await exchange.publish(
                aio_pika.Message(
                    json.dumps(message, ensure_ascii=False).encode('utf-8'),
                    content_type="application/json",
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                ),
                routing_key="document.processed"
            )
            
            logger.info(f"发布文档处理消息: {message['document_id']}")
            
        except Exception as e:
            logger.error(f"消息发布失败: {str(e)}")
            raise
```

## 📊 API端点详细规范

### 1. 文档处理API
```python
@app.post("/process-document", response_model=ProcessingResult)
async def process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_id: Optional[str] = None,
    enable_chunking: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """
    处理上传的文档
    
    参数:
        file: 上传的文档文件 (PDF, DOCX, TXT, MD)
        document_id: 可选的文档ID，不提供则自动生成UUID
        enable_chunking: 是否启用文本分块，默认True
        chunk_size: 分块大小，默认1000字符
        chunk_overlap: 分块重叠，默认200字符
    
    响应:
        ProcessingResult: 包含文档ID、状态、分块预览
    
    处理流程:
        1. 文档验证 (格式、大小)
        2. RabbitMQ消息发布 (处理开始)
        3. 后台异步处理启动
        4. 即时响应返回
        5. 异步完成向量化
        6. RabbitMQ消息发布 (处理完成)
    """
```

### 2. 处理状态查询API
```python
@app.get("/processing-status/{document_id}", response_model=ProcessingStatusResponse)
async def get_processing_status(document_id: str):
    """
    查询文档处理状态
    
    参数:
        document_id: 文档唯一标识
    
    响应:
        ProcessingStatusResponse: 处理状态、进度、时间戳
    
    状态类型:
        - processing: 处理中
        - completed: 处理完成  
        - failed: 处理失败
        - pending: 等待处理
    """
```

### 3. 健康检查API
```python
@app.get("/health")
async def health_check():
    """
    服务健康状态检查
    
    检查项:
        - 服务基本状态
        - RabbitMQ连接状态
        - Vector Service可用性
        - 文档处理器状态
    
    响应示例:
        {
            "status": "healthy",
            "service": "document-processor", 
            "version": "1.0.0",
            "components": {
                "rabbitmq": "healthy",
                "vector_service": "healthy"
            },
            "capabilities": [
                "文档上传处理",
                "多格式文档解析", 
                "文本分块",
                "异步处理通知"
            ]
        }
    """
```

## 🔄 处理流程详解

### 异步文档处理流程
```mermaid
sequenceDiagram
    participant Client as 客户端
    participant API as Document Processor API
    participant BG as 后台任务
    participant Processor as 文档处理器
    participant Vector as Vector Service
    participant MQ as RabbitMQ
    
    Client->>API: POST /process-document
    API->>API: 生成document_id
    API->>MQ: 发布处理开始消息
    API->>BG: 添加后台任务
    API->>Client: 返回处理状态
    
    BG->>Processor: 处理文档
    Processor->>Processor: 文本提取+分块
    BG->>Vector: 调用向量化服务
    Vector->>Vector: 文本向量化+存储
    Vector->>BG: 返回向量化结果
    BG->>MQ: 发布处理完成消息
```

### 错误处理和重试机制
```python
# 异步处理中的错误处理
async def process_document_async(...):
    try:
        # 核心处理逻辑
        result = await document_processor.process_file(...)
        
        # 向量化处理（带重试）
        vector_result = await retry_with_backoff(
            vector_client.vectorize_and_store,
            args=(texts, document_id),
            max_retries=3,
            base_delay=1.0
        )
        
        # 成功消息发布
        await rabbit_client.publish_document_processed({
            "status": ProcessingStatus.COMPLETED,
            "document_id": document_id,
            "chunks": chunks,
            "vectorization": vector_result
        })
        
    except DocumentProcessingError as e:
        # 文档处理特定错误
        await publish_error_message(document_id, f"文档处理失败: {str(e)}")
    except VectorizationError as e:
        # 向量化错误（文档处理成功，但向量化失败）
        await publish_partial_success(document_id, chunks, f"向量化失败: {str(e)}")
    except Exception as e:
        # 未预期的错误
        logger.error(f"文档处理异常: {str(e)}", exc_info=True)
        await publish_error_message(document_id, f"处理异常: {str(e)}")
```

## 🧪 测试和质量保证

### 单元测试
```python
# tests/test_document_processor.py
import pytest
from src.document_processor import ModernDocumentProcessor

class TestDocumentProcessor:
    @pytest.fixture
    async def processor(self):
        return ModernDocumentProcessor()
    
    async def test_pdf_processing(self, processor):
        """测试PDF文档处理"""
        with open("test_files/sample.pdf", "rb") as f:
            result = await processor.process_file(
                file_content=f.read(),
                filename="sample.pdf",
                mime_type="application/pdf",
                document_id="test-pdf-1"
            )
        
        assert result.status == "completed"
        assert len(result.chunks) > 0
        assert all(chunk.content.strip() for chunk in result.chunks)
    
    async def test_text_chunking(self, processor):
        """测试文本分块功能"""
        long_text = "测试内容。" * 500  # 创建长文本
        
        result = await processor.process_file(
            file_content=long_text.encode('utf-8'),
            filename="test.txt",
            mime_type="text/plain",
            document_id="test-chunk-1",
            chunk_size=200,
            chunk_overlap=50
        )
        
        assert result.status == "completed"
        assert len(result.chunks) > 1
        assert all(len(chunk.content) <= 250 for chunk in result.chunks)  # 允许些许超出
```

### 集成测试
```python
# tests/test_integration.py
import pytest
import httpx

class TestDocumentProcessorIntegration:
    @pytest.fixture
    def base_url(self):
        return "http://localhost:8001"
    
    async def test_full_processing_flow(self, base_url):
        """测试完整的文档处理流程"""
        async with httpx.AsyncClient() as client:
            # 1. 上传文档
            with open("test_files/test_doc.pdf", "rb") as f:
                files = {"file": f}
                data = {"enable_chunking": "true", "chunk_size": "500"}
                
                response = await client.post(
                    f"{base_url}/process-document",
                    files=files,
                    data=data
                )
                
                assert response.status_code == 200
                result = response.json()
                document_id = result["document_id"]
            
            # 2. 等待处理完成
            await asyncio.sleep(10)
            
            # 3. 检查处理状态
            status_response = await client.get(
                f"{base_url}/processing-status/{document_id}"
            )
            assert status_response.status_code == 200
```

### 性能测试
```bash
# 并发处理测试
for i in {1..10}; do
  curl -X POST "http://localhost:8001/process-document" \
    -F "file=@test_doc_$i.pdf" \
    -F "document_id=perf-test-$i" &
done
wait

# 大文件处理测试
dd if=/dev/zero of=large_test.txt bs=1M count=20
curl -X POST "http://localhost:8001/process-document" \
  -F "file=@large_test.txt" \
  -F "document_id=large-file-test"
```

## 📈 监控和运维

### 日志配置
```python
# logging_config.py - 结构化日志
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "service": "document-processor",
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'document_id'):
            log_entry["document_id"] = record.document_id
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry, ensure_ascii=False)
```

### 性能监控
```python
# 性能监控装饰器
import time
import psutil
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            logger.info(
                f"性能监控 - {func.__name__}",
                extra={
                    "duration_ms": round(duration * 1000, 2),
                    "memory_delta_mb": round(end_memory - start_memory, 2),
                    "function": func.__name__
                }
            )
            
            return result
        except Exception as e:
            logger.error(f"函数执行异常 - {func.__name__}: {str(e)}")
            raise
            
    return wrapper

# 使用示例
@monitor_performance
async def process_document_async(...):
    # 处理逻辑
    pass
```

## 🚀 部署配置

### Docker配置
```dockerfile
# Dockerfile
FROM python:3.11-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 8001

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# 启动命令
CMD ["python", "main.py"]
```

### 生产环境配置
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  document-processor:
    build: .
    environment:
      - HOST=0.0.0.0
      - PORT=8001
      - LOG_LEVEL=WARNING
      - DEBUG_MODE=false
      - RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672
      - VECTOR_SERVICE_URL=http://vector-service:8002
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 2GB
          cpus: '1.0'
        reservations:
          memory: 512MB
          cpus: '0.5'
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 📋 开发检查清单

### ✅ 服务开发规范
- [ ] 使用FastAPI + Pydantic数据验证
- [ ] 实现异步处理和BackgroundTasks
- [ ] 添加结构化JSON日志
- [ ] 实现健康检查端点
- [ ] 添加性能监控装饰器
- [ ] 遵循RESTful API设计

### ✅ 文档处理功能
- [ ] 支持多种文档格式 (PDF, DOCX, TXT, MD)
- [ ] 实现智能文本分块
- [ ] 添加文档大小和格式验证
- [ ] 实现错误处理和降级机制
- [ ] 优化大文件处理性能

### ✅ 集成和通信
- [ ] RabbitMQ消息队列集成
- [ ] Vector Service HTTP客户端
- [ ] 异步任务状态管理
- [ ] 服务间通信超时和重试
- [ ] 故障恢复机制

### ✅ 性能和监控
- [ ] 并发请求处理优化
- [ ] 内存使用监控
- [ ] 处理时间性能指标
- [ ] 错误率统计
- [ ] 健康状态监控

## 🔗 相关文档

- [Python服务集群总览](../CLAUDE.md)
- [Vector Service开发文档](../vector-service/CLAUDE.md)
- [RAG Service开发文档](../rag-service/CLAUDE.md)
- [后端API集成文档](../../backend/CLAUDE.md)
- [Docker部署配置](../../docker-compose.yml)

---

## 🚀 **RAG系统准确率优化方案**
**重要**: Document Processor作为文档处理入口，正在实施[RAG系统准确率优化方案](../../CLAUDE.md#rag系统准确率优化方案)，目标将准确率从53.4%-65%提升至90-95%。

### 📋 Document Processor优化重点

#### ✅ Phase 1: 语义分块优化 (进行中)
**智能文档分块** - 替换当前固定长度分块
```python
class SemanticChunker:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        self.similarity_threshold = 0.5
        
    async def semantic_chunking(self, text: str, max_chunk_size: int = 1000):
        """基于语义相似度的智能分块"""
        sentences = self.split_sentences(text)
        embeddings = await self.embed_sentences(sentences)
        
        chunks = []
        current_chunk = []
        current_embedding = None
        
        for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
            if current_embedding is None:
                current_chunk.append(sentence)
                current_embedding = embedding
            else:
                similarity = cosine_similarity([current_embedding], [embedding])[0][0]
                
                if similarity >= self.similarity_threshold and len(" ".join(current_chunk)) < max_chunk_size:
                    current_chunk.append(sentence)
                    # 更新当前chunk的代表性嵌入
                    current_embedding = np.mean([current_embedding, embedding], axis=0)
                else:
                    # 开始新的chunk
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_embedding = embedding
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
```

**增强元数据提取** - 提升检索精度
```python
class EnhancedMetadataExtractor:
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.keyword_extractor = KeywordExtractor()
        
    async def extract_enhanced_metadata(self, text: str, document_info: dict):
        """提取增强元数据"""
        metadata = {
            "document_id": document_info.get("document_id"),
            "filename": document_info.get("filename"),
            "mime_type": document_info.get("mime_type"),
            "timestamp": datetime.now().isoformat()
        }
        
        # 实体提取
        entities = await self.entity_extractor.extract(text)
        metadata["entities"] = entities
        
        # 关键词提取
        keywords = await self.keyword_extractor.extract(text, top_k=10)
        metadata["keywords"] = keywords
        
        # 文档结构信息
        metadata["structure"] = self.analyze_structure(text)
        
        # 文档质量评分
        metadata["quality_score"] = self.calculate_quality_score(text)
        
        return metadata
```

#### 🔄 Phase 2: 文档预处理优化 (计划中)
**多模态文档处理** - 表格和图像内容提取
```python
class MultiModalProcessor:
    def __init__(self):
        self.table_extractor = TableExtractor()
        self.image_analyzer = ImageAnalyzer()
        
    async def process_multimodal_content(self, file_content: bytes, mime_type: str):
        """处理多模态文档内容"""
        results = {
            "text": "",
            "tables": [],
            "images": [],
            "metadata": {}
        }
        
        if mime_type == "application/pdf":
            # 提取文本
            text = await self.extract_text(file_content)
            results["text"] = text
            
            # 提取表格
            tables = await self.table_extractor.extract_tables(file_content)
            results["tables"] = tables
            
            # 提取图像描述
            images = await self.image_analyzer.analyze_images(file_content)
            results["images"] = images
            
        return results
```

#### 📈 Phase 3: 知识图谱集成 (规划中)
**文档实体链接** - 知识图谱增强
```python
class DocumentGraphProcessor:
    def __init__(self, graph_client):
        self.graph_client = graph_client
        self.entity_linker = EntityLinker()
        
    async def process_with_graph(self, chunks: List[str], metadata: dict):
        """基于知识图谱增强文档处理"""
        enhanced_chunks = []
        
        for chunk in chunks:
            # 实体链接
            entities = await self.entity_linker.link_entities(chunk)
            
            # 图谱增强
            graph_context = await self.graph_client.get_context(entities)
            
            # 增强chunk内容
            enhanced_chunk = {
                "text": chunk,
                "entities": entities,
                "graph_context": graph_context,
                "metadata": metadata
            }
            enhanced_chunks.append(enhanced_chunk)
            
        return enhanced_chunks
```

### 🔧 开发优先级
1. **立即实施**: 语义分块替换固定长度分块
2. **本周实施**: 增强元数据提取和文档质量评分
3. **下周实施**: 多模态内容处理集成
4. **月内完成**: 知识图谱集成和实体链接

### 📊 预期效果
- **Phase 1完成**: 文档分块质量提升30%，检索准确率提升至70%
- **Phase 2完成**: 多模态内容支持，结构化数据提取质量提升50%
- **Phase 3完成**: 知识图谱增强，实体相关查询准确率提升至90%

详细技术方案和实施路线图请参考[根目录RAG优化方案](../../CLAUDE.md#rag系统准确率优化方案)。

---

**维护说明**: 本文档是Document Processor Service的开发指导，包含完整的架构设计、开发流程和最佳实践。配合[Python服务集群总文档](../CLAUDE.md)和[根目录优化方案](../../CLAUDE.md)，确保Document Processor在RAG准确率优化过程中的基础作用得到充分发挥。