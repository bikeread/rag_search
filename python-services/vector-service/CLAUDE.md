# Vector Service - 开发指导文档

## 📋 服务概览

**Vector Service** 是RAG系统的向量化微服务，负责文本向量化、向量存储和相似性搜索。采用CPU友好的TF-IDF向量化器，集成Milvus向量数据库，提供高性能的向量处理能力，支持中文文本和384维固定向量格式。

### 🏗️ 技术栈
- **框架**: FastAPI + Uvicorn + Pydantic
- **向量化**: TF-IDF (scikit-learn) + NumPy
- **向量数据库**: Milvus + pymilvus
- **数据处理**: NumPy, sklearn.preprocessing
- **配置管理**: Pydantic Settings
- **端口**: 8002

### 📁 项目结构
```
vector-service/
├── main.py                    # FastAPI应用和路由
├── config.py                  # Pydantic配置管理
├── milvus_client.py          # Milvus数据库客户端
├── logging_config.py          # 日志配置
├── utils/                     # 工具库
│   ├── exceptions.py         # 异常定义
│   ├── validators.py         # 输入验证
│   └── logging_utils.py      # 日志工具
└── requirements.txt          # Python依赖
```

## 🎯 核心功能模块

### 🔢 1. 文本向量化 (`POST /vectorize`)
- **TF-IDF向量化**: CPU友好，无GPU依赖
- **384维固定输出**: 便于存储和检索
- **批量处理**: 支持多文本同时向量化
- **中文文本支持**: 优化的中文处理pipeline

### 🔍 2. 向量搜索 (`POST /search`)
- **语义相似性搜索**: 基于向量余弦相似度
- **文本查询**: 自动向量化查询文本
- **向量查询**: 支持直接向量输入
- **Top-K检索**: 可配置返回结果数量

### 💾 3. 向量存储 (`POST /store-vectors`)
- **Milvus集成**: 高性能向量数据库
- **批量存储**: 优化的批量向量插入
- **元数据关联**: 向量与文本、文档ID关联
- **自动索引**: 支持COSINE相似度索引

## 🧪 开发和调试指南

### 环境配置
```bash
# 核心服务配置
HOST=0.0.0.0
PORT=8002
LOG_LEVEL=INFO
DEBUG_MODE=true

# Milvus配置
MILVUS_HOST=milvus
MILVUS_PORT=19530
MILVUS_COLLECTION=rag_documents
MILVUS_TIMEOUT=30

# 向量化配置
VECTOR_DIM=384
DEFAULT_TOP_K=5
MAX_BATCH_SIZE=100

# 性能配置
MAX_CONCURRENT_REQUESTS=20
REQUEST_TIMEOUT=30
ENABLE_CACHE=true
CACHE_TTL=3600
```

### 本地开发启动
```bash
# 1. 安装依赖
pip install -r requirements.txt
pip install -r ../shared/requirements.txt

# 2. 启动依赖服务
docker-compose up -d milvus etcd minio

# 3. 启动开发服务器
python main.py

# 4. 验证服务状态
curl http://localhost:8002/health

# 服务地址: http://localhost:8002
# API文档: http://localhost:8002/docs
```

### Docker开发模式
```bash
# 构建并启动服务
docker-compose up -d vector-service

# 查看服务日志
docker-compose logs -f vector-service

# 重启服务
docker-compose restart vector-service

# 进入容器调试
docker-compose exec vector-service bash
```

## 🔧 核心组件详解

### 1. CPU友好向量化器
```python
# main.py - CPUVectorizer类
class CPUVectorizer:
    """CPU友好的轻量级向量化器，适用于AMD处理器"""
    
    def __init__(self):
        # 使用TF-IDF作为基础向量化方法
        self.tfidf = TfidfVectorizer(
            max_features=384,       # 固定维度，便于存储
            stop_words=None,        # 保留所有词，包括中文
            ngram_range=(1, 2),     # 包含单词和双词
            min_df=1,               # 最少出现1次
            max_df=0.95,            # 忽略高频词
            token_pattern=r'[^\s]+', # 支持中文分词
            lowercase=True,         # 统一小写
            sublinear_tf=True       # 使用对数TF
        )
        self.target_dim = 384
        self.is_fitted = False
    
    def vectorize_texts(self, texts: List[str]) -> np.ndarray:
        """将文本转换为384维向量"""
        # 1. 文本预处理
        processed_texts = []
        for text in texts:
            if not text or not text.strip():
                processed_texts.append("empty_text_placeholder")
            else:
                processed_texts.append(text.strip())
        
        # 2. TF-IDF向量化
        if not self.is_fitted:
            vectors = self.tfidf.fit_transform(processed_texts)
            self.is_fitted = True
        else:
            vectors = self.tfidf.transform(processed_texts)
        
        # 3. 转换为密集数组
        dense_vectors = vectors.toarray()
        
        # 4. 维度调整到384
        current_dim = dense_vectors.shape[1]
        if current_dim < self.target_dim:
            # 维度不足时，添加小随机数填充
            padding = np.random.normal(0, 0.01, 
                (dense_vectors.shape[0], self.target_dim - current_dim))
            dense_vectors = np.hstack([dense_vectors, padding])
        elif current_dim > self.target_dim:
            # 维度过多时，截取前384维
            dense_vectors = dense_vectors[:, :self.target_dim]
        
        # 5. 防止全零向量
        for i, vec in enumerate(dense_vectors):
            if np.allclose(vec, 0):
                dense_vectors[i] = np.random.normal(0, 0.01, self.target_dim)
        
        # 6. L2归一化
        normalized_vectors = normalize(dense_vectors, norm='l2')
        
        return normalized_vectors
```

### 2. Milvus客户端集成
```python
# milvus_client.py - Milvus数据库客户端
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

class MilvusClient:
    def __init__(self):
        self.host = settings.milvus_host
        self.port = settings.milvus_port
        self.collection_name = settings.milvus_collection
        self.collection = None
        self.is_connected = False
    
    async def connect(self):
        """连接Milvus数据库"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                timeout=30
            )
            await self._setup_collection()
            self.is_connected = True
            logger.info("Milvus连接成功")
        except Exception as e:
            logger.error(f"Milvus连接失败: {str(e)}")
            raise
    
    async def _setup_collection(self):
        """初始化向量集合"""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, 
                       max_length=100, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, 
                       dim=384),
            FieldSchema(name="text", dtype=DataType.VARCHAR, 
                       max_length=65535),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, 
                       max_length=100),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        
        schema = CollectionSchema(fields, description="RAG文档向量存储")
        
        if not Collection.exists(self.collection_name):
            collection = Collection(self.collection_name, schema)
            
            # 创建COSINE相似度索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT", 
                "params": {"nlist": 1024}
            }
            collection.create_index("vector", index_params)
            logger.info("创建新的向量集合和索引")
        
        self.collection = Collection(self.collection_name)
        self.collection.load()
    
    async def upsert_vectors(
        self, 
        vectors: List[List[float]], 
        texts: List[str],
        document_id: str,
        metadata: List[Dict[str, Any]] = None
    ):
        """批量插入或更新向量"""
        if not self.is_connected:
            raise RuntimeError("Milvus未连接")
        
        # 生成向量ID
        vector_ids = [f"{document_id}-{i}" for i in range(len(vectors))]
        
        # 准备插入数据
        entities = [
            vector_ids,
            vectors,
            texts,
            [document_id] * len(vectors),
            metadata or [{} for _ in vectors]
        ]
        
        # 批量插入
        result = self.collection.insert(entities)
        self.collection.flush()
        
        logger.info(f"成功插入 {len(vectors)} 个向量")
        return result
    
    async def search_vectors(
        self, 
        query_vector: List[float], 
        top_k: int = 5,
        filters: str = None
    ) -> List[Dict[str, Any]]:
        """向量相似性搜索"""
        if not self.is_connected:
            return []
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=["text", "document_id", "metadata"],
            expr=filters
        )
        
        # 格式化搜索结果
        formatted_results = []
        for hit in results[0]:
            formatted_results.append({
                "id": hit.id,
                "score": hit.score,
                "text": hit.entity.get("text"),
                "document_id": hit.entity.get("document_id"),
                "metadata": hit.entity.get("metadata", {})
            })
        
        return formatted_results
```

### 3. 异步API端点设计
```python
# main.py - API端点实现
@app.post("/vectorize", response_model=VectorizeResponse)
async def vectorize_texts(request: VectorizeRequest):
    """向量化文本并可选存储到Milvus"""
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        logger.info(f"开始向量化 {len(request.texts)} 个文本")
        
        # 1. 文本向量化
        vectors = vectorizer.vectorize_texts(request.texts)
        
        # 2. 生成向量ID
        vector_ids = request.document_ids or [str(uuid.uuid4()) for _ in request.texts]
        
        # 3. 格式化向量数据
        vector_list = [vec.tolist() for vec in vectors]
        
        stored_count = 0
        
        # 4. 存储到Milvus（可选）
        if request.store_vectors and milvus_client and milvus_client.is_connected:
            try:
                # 准备元数据
                metadata_list = request.metadata or [{} for _ in request.texts]
                for i, meta in enumerate(metadata_list):
                    meta.update({
                        "text": request.texts[i],
                        "vector_id": vector_ids[i],
                        "created_at": datetime.now().isoformat()
                    })
                
                # 批量存储
                document_id = (request.document_ids[0] if request.document_ids 
                             else f"batch_{str(uuid.uuid4())[:8]}")
                
                await milvus_client.upsert_vectors(
                    vectors=vector_list,
                    texts=request.texts,
                    document_id=document_id,
                    metadata=metadata_list
                )
                stored_count = len(vector_list)
                logger.info(f"成功存储 {stored_count} 个向量")
                
            except Exception as e:
                logger.error(f"向量存储失败: {str(e)}")
                # 继续执行，允许返回向量但不存储
        
        return VectorizeResponse(
            vector_ids=vector_ids,
            vectors=vector_list,
            status="completed",
            stored_count=stored_count
        )
        
    except Exception as e:
        logger.error(f"向量化失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"向量化失败: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_vectors(request: SearchRequest):
    """向量相似性搜索"""
    import time
    start_time = time.time()
    
    try:
        # 1. 获取查询向量
        if request.query_text:
            # 从文本生成查询向量
            vectors = vectorizer.vectorize_texts([request.query_text])
            query_vector = vectors[0].tolist()
        elif request.query_vector:
            # 使用提供的查询向量
            query_vector = request.query_vector
        else:
            raise HTTPException(status_code=400, 
                detail="必须提供query_text或query_vector")
        
        results = []
        
        # 2. 执行Milvus搜索
        if milvus_client and milvus_client.is_connected:
            try:
                search_results = await milvus_client.search_vectors(
                    query_vector=query_vector,
                    top_k=request.top_k,
                    filters=request.filter_expr
                )
                
                # 格式化搜索结果
                for result in search_results:
                    results.append({
                        "id": result.get("id"),
                        "score": result.get("score", 0.0),
                        "text": result.get("text", ""),
                        "document_id": result.get("document_id", ""),
                        "metadata": result.get("metadata", {})
                    })
                
                logger.info(f"搜索完成，找到 {len(results)} 个结果")
                
            except Exception as e:
                logger.error(f"向量搜索失败: {str(e)}")
        else:
            logger.warning("Milvus未连接，无法执行搜索")
        
        search_time = time.time() - start_time
        
        return SearchResponse(
            results=results,
            query_vector=query_vector,
            status="completed", 
            search_time=search_time
        )
        
    except Exception as e:
        logger.error(f"搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")
```

## 📊 API端点详细规范

### 1. 文本向量化API
```python
POST /vectorize
Content-Type: application/json

{
    "texts": ["文本1", "文本2", "文本3"],
    "document_ids": ["doc-1", "doc-2", "doc-3"],  // 可选
    "metadata": [
        {"source": "pdf", "page": 1},
        {"source": "pdf", "page": 2},
        {"source": "pdf", "page": 3}
    ],  // 可选
    "store_vectors": true,  // 是否存储到Milvus
    "collection_name": "rag_documents"  // 集合名称
}

响应:
{
    "vector_ids": ["vec-id-1", "vec-id-2", "vec-id-3"],
    "vectors": [
        [0.1, 0.2, ..., 0.384],  // 384维向量
        [0.3, 0.4, ..., 0.384],
        [0.5, 0.6, ..., 0.384]
    ],
    "status": "completed",
    "stored_count": 3
}
```

### 2. 向量搜索API
```python
POST /search
Content-Type: application/json

{
    "query_text": "搜索关键词",  // 文本查询 (二选一)
    "query_vector": [0.1, 0.2, ..., 0.384],  // 向量查询 (二选一)
    "top_k": 5,  // 返回结果数量
    "collection_name": "rag_documents",
    "filter_expr": "document_id == 'doc-123'"  // 可选过滤条件
}

响应:
{
    "results": [
        {
            "id": "doc-123-chunk-0",
            "score": 0.95,  // 相似度分数 (0-1)
            "text": "匹配的文本内容...",
            "document_id": "doc-123",
            "metadata": {"source": "pdf", "page": 1}
        }
    ],
    "query_vector": [0.1, 0.2, ..., 0.384],
    "status": "completed",
    "search_time": 0.123  // 搜索耗时(秒)
}
```

### 3. 直接存储向量API
```python
POST /store-vectors
Content-Type: application/json

{
    "vectors": [
        [0.1, 0.2, ..., 0.384],
        [0.3, 0.4, ..., 0.384]
    ],
    "vector_ids": ["ext-vec-1", "ext-vec-2"],
    "metadata": [
        {"source": "external", "type": "embedding"},
        {"source": "external", "type": "embedding"}
    ],
    "collection_name": "rag_documents"
}

响应:
{
    "status": "completed",
    "stored_count": 2,
    "collection_name": "rag_documents"
}
```

## 🧪 测试和质量保证

### 单元测试
```python
# tests/test_vectorizer.py
import pytest
import numpy as np
from main import CPUVectorizer

class TestCPUVectorizer:
    @pytest.fixture
    def vectorizer(self):
        return CPUVectorizer()
    
    def test_single_text_vectorization(self, vectorizer):
        """测试单文本向量化"""
        texts = ["这是一个测试文档"]
        vectors = vectorizer.vectorize_texts(texts)
        
        assert vectors.shape == (1, 384)
        assert np.allclose(np.linalg.norm(vectors[0]), 1.0)  # 验证L2归一化
    
    def test_batch_vectorization(self, vectorizer):
        """测试批量向量化"""
        texts = ["文档一", "文档二", "文档三"]
        vectors = vectorizer.vectorize_texts(texts)
        
        assert vectors.shape == (3, 384)
        # 验证每个向量都已归一化
        for vec in vectors:
            assert np.allclose(np.linalg.norm(vec), 1.0)
    
    def test_chinese_text_support(self, vectorizer):
        """测试中文文本支持"""
        chinese_texts = [
            "这是中文测试文档，包含常用词汇。",
            "测试中文分词效果和向量生成质量。"
        ]
        vectors = vectorizer.vectorize_texts(chinese_texts)
        
        assert vectors.shape == (2, 384)
        # 验证两个向量不完全相同（内容不同）
        assert not np.allclose(vectors[0], vectors[1])
    
    def test_empty_text_handling(self, vectorizer):
        """测试空文本处理"""
        texts = ["", "  ", "有效文本"]
        vectors = vectorizer.vectorize_texts(texts)
        
        assert vectors.shape == (3, 384)
        # 验证空文本也产生了有效向量
        assert not np.all(vectors[0] == 0)
        assert not np.all(vectors[1] == 0)
        assert not np.all(vectors[2] == 0)
```

### 集成测试
```python
# tests/test_integration.py
import pytest
import httpx
import asyncio

class TestVectorServiceIntegration:
    @pytest.fixture
    def base_url(self):
        return "http://localhost:8002"
    
    async def test_vectorize_and_search_flow(self, base_url):
        """测试向量化和搜索完整流程"""
        async with httpx.AsyncClient() as client:
            # 1. 向量化文本
            vectorize_response = await client.post(
                f"{base_url}/vectorize",
                json={
                    "texts": ["机器学习是人工智能的分支", "深度学习使用神经网络"],
                    "store_vectors": True
                }
            )
            assert vectorize_response.status_code == 200
            vectorize_data = vectorize_response.json()
            assert vectorize_data["status"] == "completed"
            assert len(vectorize_data["vectors"]) == 2
            
            # 2. 等待存储完成
            await asyncio.sleep(2)
            
            # 3. 执行相似性搜索
            search_response = await client.post(
                f"{base_url}/search",
                json={
                    "query_text": "人工智能技术",
                    "top_k": 2
                }
            )
            assert search_response.status_code == 200
            search_data = search_response.json()
            assert search_data["status"] == "completed"
            assert len(search_data["results"]) > 0
    
    async def test_milvus_integration(self, base_url):
        """测试Milvus集成"""
        async with httpx.AsyncClient() as client:
            # 健康检查
            health_response = await client.get(f"{base_url}/health")
            health_data = health_response.json()
            
            if health_data["components"]["milvus"] == "connected":
                # 测试直接向量存储
                store_response = await client.post(
                    f"{base_url}/store-vectors",
                    json={
                        "vectors": [[0.1] * 384, [0.2] * 384],
                        "vector_ids": ["test-1", "test-2"],
                        "metadata": [{"type": "test"}, {"type": "test"}]
                    }
                )
                assert store_response.status_code == 200
```

### 性能测试
```python
# tests/test_performance.py
import time
import asyncio
import httpx

async def test_vectorization_performance():
    """测试向量化性能"""
    # 生成测试数据
    texts = [f"测试文档 {i}: " + "内容 " * 50 for i in range(100)]
    
    async with httpx.AsyncClient() as client:
        start_time = time.time()
        
        response = await client.post(
            "http://localhost:8002/vectorize",
            json={"texts": texts, "store_vectors": False},
            timeout=30.0
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["vectors"]) == 100
        assert duration < 10.0  # 100个文档应在10秒内完成
        
        print(f"向量化100个文档耗时: {duration:.2f}秒")

async def test_search_performance():
    """测试搜索性能"""
    async with httpx.AsyncClient() as client:
        # 先插入测试数据
        await client.post(
            "http://localhost:8002/vectorize",
            json={
                "texts": ["AI", "ML", "DL", "NLP", "CV"] * 20,
                "store_vectors": True
            }
        )
        
        # 等待存储完成
        await asyncio.sleep(2)
        
        # 测试搜索性能
        start_time = time.time()
        
        response = await client.post(
            "http://localhost:8002/search",
            json={"query_text": "人工智能技术", "top_k": 10}
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert response.status_code == 200
        assert duration < 1.0  # 搜索应在1秒内完成
        
        print(f"向量搜索耗时: {duration:.3f}秒")
```

## 📈 监控和运维

### 性能监控指标
```python
# 自定义性能监控中间件
import time
import psutil
from fastapi import Request, Response

@app.middleware("http")
async def performance_monitoring(request: Request, call_next):
    """性能监控中间件"""
    start_time = time.time()
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # 执行请求
    response = await call_next(request)
    
    # 计算性能指标
    duration = time.time() - start_time
    end_memory = process.memory_info().rss / 1024 / 1024
    memory_delta = end_memory - start_memory
    
    # 记录性能日志
    logger.info(
        f"API性能监控",
        extra={
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 2),
            "memory_delta_mb": round(memory_delta, 2),
            "cpu_percent": process.cpu_percent()
        }
    )
    
    # 添加响应头
    response.headers["X-Process-Time"] = str(duration)
    
    return response

# 健康检查增强
@app.get("/metrics")
async def get_metrics():
    """获取服务性能指标"""
    process = psutil.Process()
    
    return {
        "service": "vector-service",
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "cpu_percent": process.cpu_percent(),
            "memory_mb": round(process.memory_info().rss / 1024 / 1024, 2),
            "vectorizer_fitted": vectorizer.is_fitted,
            "milvus_connected": milvus_client.is_connected if milvus_client else False
        },
        "counters": {
            "vectorize_requests": getattr(app.state, "vectorize_count", 0),
            "search_requests": getattr(app.state, "search_count", 0),
            "store_requests": getattr(app.state, "store_count", 0)
        }
    }
```

### 日志配置优化
```python
# logging_config.py - 结构化日志
import logging
import json
from datetime import datetime

class VectorServiceFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "service": "vector-service",
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName
        }
        
        # 添加向量化相关上下文
        if hasattr(record, 'vector_count'):
            log_entry["vector_count"] = record.vector_count
        if hasattr(record, 'search_time'):
            log_entry["search_time"] = record.search_time
        if hasattr(record, 'milvus_operation'):
            log_entry["milvus_operation"] = record.milvus_operation
            
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry, ensure_ascii=False)
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
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建非root用户
RUN adduser --disabled-password --gecos '' --uid 1000 vectoruser
USER vectoruser

# 暴露端口
EXPOSE 8002

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8002/health || exit 1

# 启动命令
CMD ["python", "main.py"]
```

### 生产环境配置
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  vector-service:
    build: .
    environment:
      - HOST=0.0.0.0
      - PORT=8002
      - LOG_LEVEL=WARNING
      - MILVUS_HOST=milvus
      - MILVUS_PORT=19530
      - VECTOR_DIM=384
      - MAX_BATCH_SIZE=50
      - ENABLE_CACHE=true
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 3GB
          cpus: '1.5'
        reservations:
          memory: 1GB
          cpus: '0.5'
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    depends_on:
      - milvus
```

### Kubernetes部署
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vector-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vector-service
  template:
    metadata:
      labels:
        app: vector-service
    spec:
      containers:
      - name: vector-service
        image: vector-service:latest
        ports:
        - containerPort: 8002
        env:
        - name: MILVUS_HOST
          value: "milvus-service"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "3Gi" 
            cpu: "1500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8002
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8002
          initialDelaySeconds: 5
          periodSeconds: 5
```

## 📋 开发检查清单

### ✅ 向量化功能
- [ ] TF-IDF向量化器正常工作
- [ ] 384维向量输出
- [ ] L2归一化实现
- [ ] 中文文本支持
- [ ] 批量处理优化
- [ ] 空文本处理

### ✅ 数据库集成
- [ ] Milvus连接和断开
- [ ] 向量集合自动创建
- [ ] COSINE索引配置
- [ ] 批量向量插入
- [ ] 元数据关联存储
- [ ] 连接故障降级

### ✅ 搜索功能
- [ ] 文本查询向量化
- [ ] 向量相似性搜索
- [ ] Top-K结果返回
- [ ] 相似度分数计算
- [ ] 过滤条件支持
- [ ] 搜索性能优化

### ✅ API和服务
- [ ] FastAPI规范实现
- [ ] Pydantic数据验证
- [ ] 异常处理机制
- [ ] 性能监控埋点
- [ ] 结构化日志记录
- [ ] 健康检查端点

## 🔗 相关文档

- [Python服务集群总览](../CLAUDE.md)
- [Document Processor开发文档](../document-processor/CLAUDE.md)
- [RAG Service开发文档](../rag-service/CLAUDE.md)
- [后端API集成文档](../../backend/CLAUDE.md)
- [Docker部署配置](../../docker-compose.yml)

---

## 🚀 **RAG系统准确率优化方案**
**重要**: Vector Service作为向量化引擎，正在实施[RAG系统准确率优化方案](../../CLAUDE.md#rag系统准确率优化方案)，目标将准确率从53.4%-65%提升至90-95%。

### 📋 Vector Service优化重点

#### ✅ Phase 1: 混合检索系统 (进行中)
**BM25检索器集成** - 精确匹配增强
```python
class BM25Retriever:
    def __init__(self):
        self.index = None
        self.corpus = []
        self.tokenizer = jieba.lcut  # 中文分词
        
    def build_index(self, documents: List[str]):
        """构建BM25索引"""
        tokenized_corpus = [self.tokenizer(doc) for doc in documents]
        self.corpus = documents
        self.index = BM25Okapi(tokenized_corpus)
        
    async def search(self, query: str, top_k: int = 10):
        """BM25检索"""
        if not self.index:
            return []
            
        tokenized_query = self.tokenizer(query)
        scores = self.index.get_scores(tokenized_query)
        
        # 获取top-k结果
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回有匹配的结果
                results.append({
                    "id": f"bm25_{idx}",
                    "score": float(scores[idx]),
                    "text": self.corpus[idx],
                    "retrieval_type": "bm25"
                })
                
        return results
```

**RRF融合机制** - 多检索器结果融合
```python
class RecipRankFusion:
    def __init__(self, k: int = 60):
        self.k = k  # RRF参数
        
    def fuse_results(self, *result_lists):
        """融合多个检索器的结果"""
        doc_scores = {}
        
        for results in result_lists:
            for rank, doc in enumerate(results):
                doc_id = doc.get("id", "")
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "rrf_score": 0.0,
                        "doc": doc,
                        "source_scores": {}
                    }
                
                # RRF计算: 1/(k + rank)
                rrf_contribution = 1.0 / (self.k + rank + 1)
                doc_scores[doc_id]["rrf_score"] += rrf_contribution
                
                # 记录来源分数
                retrieval_type = doc.get("retrieval_type", "unknown")
                doc_scores[doc_id]["source_scores"][retrieval_type] = doc.get("score", 0.0)
        
        # 按RRF分数排序
        fused_results = sorted(
            doc_scores.values(), 
            key=lambda x: x["rrf_score"], 
            reverse=True
        )
        
        return [item["doc"] for item in fused_results]
```

**向量索引优化** - 提升检索性能
```python
class OptimizedVectorIndex:
    def __init__(self, vector_dim: int = 384):
        self.vector_dim = vector_dim
        self.index_type = "IVF_PQ"  # 量化索引，节省内存
        
    async def create_optimized_index(self, collection_name: str):
        """创建优化的向量索引"""
        # IVF_PQ: 倒排文件 + 乘积量化
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_PQ",
            "params": {
                "nlist": 1024,    # 聚类中心数
                "m": 8,           # PQ子空间数
                "nbits": 8        # 每个子空间的位数
            }
        }
        
        collection = Collection(collection_name)
        collection.create_index("vector", index_params)
        collection.load()
        
        return collection
```

#### 🔄 Phase 2: 嵌入模型优化 (计划中)
**嵌入模型微调** - 领域适应
```python
class EmbeddingFineTuner:
    def __init__(self, base_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.base_model = base_model
        self.fine_tuned_model = None
        
    async def fine_tune_with_lora(self, training_pairs: List[Tuple[str, str]]):
        """使用LoRA微调嵌入模型"""
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.losses import CosineSimilarityLoss
        
        # 加载基础模型
        model = SentenceTransformer(self.base_model)
        
        # 构造训练数据
        train_examples = []
        for query, passage in training_pairs:
            train_examples.append(InputExample(texts=[query, passage], label=1.0))
            
        # 训练数据加载器
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        
        # 损失函数
        train_loss = CosineSimilarityLoss(model)
        
        # 微调训练
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=3,
            warmup_steps=100,
            optimizer_params={'lr': 2e-5}
        )
        
        self.fine_tuned_model = model
        return model
```

**合成数据生成** - 扩充训练数据
```python
class SyntheticDataGenerator:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        
    async def generate_query_passage_pairs(self, documents: List[str], num_pairs: int = 1000):
        """生成查询-文档对"""
        training_pairs = []
        
        for doc in documents[:num_pairs // 10]:  # 每个文档生成10对
            prompt = f"""
            基于以下文档内容，生成10个相关的查询问题，确保问题能通过文档内容回答：
            
            文档：{doc[:500]}...
            
            请以JSON格式返回：
            {{"queries": ["问题1", "问题2", ...]}}
            """
            
            response = await self.llm_client.generate(prompt)
            queries = self.parse_queries(response)
            
            for query in queries:
                training_pairs.append((query, doc))
                
        return training_pairs
```

#### 📈 Phase 3: 高级向量增强 (规划中)
**多向量表示** - 密集+稀疏混合
```python
class HybridVectorizer:
    def __init__(self):
        self.dense_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.sparse_model = TfidfVectorizer(max_features=1000)
        
    async def create_hybrid_vectors(self, texts: List[str]):
        """创建混合向量表示"""
        # 密集向量
        dense_vectors = self.dense_model.encode(texts)
        
        # 稀疏向量
        sparse_vectors = self.sparse_model.fit_transform(texts).toarray()
        
        # 拼接混合向量
        hybrid_vectors = np.concatenate([dense_vectors, sparse_vectors], axis=1)
        
        return hybrid_vectors
```

**向量压缩和量化** - 存储优化
```python
class VectorCompressor:
    def __init__(self, compression_ratio: float = 0.5):
        self.compression_ratio = compression_ratio
        self.pca = None
        
    def compress_vectors(self, vectors: np.ndarray):
        """使用PCA压缩向量"""
        original_dim = vectors.shape[1]
        target_dim = int(original_dim * self.compression_ratio)
        
        self.pca = PCA(n_components=target_dim)
        compressed_vectors = self.pca.fit_transform(vectors)
        
        return compressed_vectors
```

### 🔧 开发优先级
1. **立即实施**: BM25检索器集成和RRF融合机制
2. **本周实施**: 向量索引优化和检索性能提升
3. **下周实施**: 嵌入模型微调和合成数据生成
4. **月内完成**: 多向量表示和向量压缩优化

### 📊 预期效果
- **Phase 1完成**: 混合检索准确率提升25%，精确匹配能力增强
- **Phase 2完成**: 领域适应性提升，语义理解准确率提升30%
- **Phase 3完成**: 存储效率提升50%，检索速度提升40%

### 🧪 性能验证
```python
async def benchmark_retrieval_quality():
    """检索质量基准测试"""
    test_queries = load_test_queries()
    ground_truth = load_ground_truth()
    
    # 测试不同检索器
    bm25_results = await bm25_retriever.batch_search(test_queries)
    vector_results = await vector_retriever.batch_search(test_queries)
    hybrid_results = await hybrid_retriever.batch_search(test_queries)
    
    # 计算评估指标
    bm25_metrics = calculate_metrics(bm25_results, ground_truth)
    vector_metrics = calculate_metrics(vector_results, ground_truth)
    hybrid_metrics = calculate_metrics(hybrid_results, ground_truth)
    
    return {
        "bm25": bm25_metrics,
        "vector": vector_metrics,
        "hybrid": hybrid_metrics
    }
```

详细技术方案和实施路线图请参考[根目录RAG优化方案](../../CLAUDE.md#rag系统准确率优化方案)。

---

**维护说明**: 本文档是Vector Service的开发指导，包含完整的架构设计、开发流程和最佳实践。配合[Python服务集群总文档](../CLAUDE.md)和[根目录优化方案](../../CLAUDE.md)，确保Vector Service在RAG准确率优化过程中的核心作用得到充分发挥。