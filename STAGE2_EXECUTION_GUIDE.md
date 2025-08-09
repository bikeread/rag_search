# 阶段2执行指导文档：Python服务重构

## 📋 总览信息

**目标**: 将现有Python代码重构为专业化微服务架构
**预计时间**: 14-17天 (2-2.5周，使用并行开发策略)
**执行策略**: 方案B - 并行开发 (高效型)

## 🎯 核心目标

- ✅ **文档处理服务** (端口8001) - 处理文档上传和分块
- ✅ **向量化服务** (端口8002) - 文本向量化和存储  
- ✅ **RAG检索服务** (端口8003) - 语义检索和答案生成
- ✅ **共享基础设施** - Milvus客户端、消息队列、工具库

---

## 🏗️ 子阶段拆分执行计划

### 子阶段2.1：基础设施准备 (3-4天) 🔧

**状态**: 🔴 待开始  
**优先级**: 🔥 最高 (阻塞其他任务)  
**依赖**: 阶段1完成  

#### 📁 目录结构创建

```bash
python-services/
├── shared/                          # 共享组件
│   ├── __init__.py
│   ├── milvus_client.py             # Milvus客户端
│   ├── rabbit_client.py             # RabbitMQ客户端  
│   ├── config.py                    # 配置管理
│   ├── logging_config.py            # 日志配置
│   └── utils/
│       ├── __init__.py
│       ├── exceptions.py            # 自定义异常
│       └── validators.py            # 数据验证
├── document-processor/              # 文档处理服务
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                      # FastAPI入口
│   └── src/
│       ├── __init__.py
│       ├── processors/              # 从现有代码迁移
│       ├── splitting/               # 从现有代码迁移  
│       └── document_processor.py    # 主处理逻辑
├── vector-service/                  # 向量化服务
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                      # FastAPI入口
│   └── src/
│       ├── __init__.py
│       ├── vectorizer.py            # 向量化器
│       └── vector_manager.py        # 向量管理
└── rag-service/                     # RAG检索服务
    ├── Dockerfile
    ├── requirements.txt
    ├── main.py                      # FastAPI入口
    └── src/
        ├── __init__.py
        ├── retriever.py             # 检索器
        ├── llm_client.py            # LLM客户端
        └── rag_processor.py         # RAG处理器
```

#### 🎯 具体任务清单

**Task 2.1.1: Milvus客户端重构**

```python
# 文件: python-services/shared/milvus_client.py
# 基于现有 src/data_processing/storage/milvus_store.py

class MilvusClient:
    """统一的Milvus客户端，供所有服务使用"""
    
    def __init__(self, collection_name: str = "document_store"):
        # 连接配置从环境变量获取
        
    async def connect(self) -> bool:
        # 建立连接并验证
        
    async def create_collection_if_not_exists(self, dim: int = 384):
        # 创建集合和索引
        
    async def upsert_vectors(self, 
                           vectors: List[List[float]], 
                           texts: List[str], 
                           metadata: List[Dict] = None) -> List[str]:
        # 批量插入向量
        
    async def search_vectors(self, 
                           query_vector: List[float], 
                           top_k: int = 5,
                           filters: Dict = None) -> List[Dict]:
        # 向量检索
        
    async def delete_vectors(self, vector_ids: List[str]) -> bool:
        # 删除向量
        
    async def get_collection_stats(self) -> Dict:
        # 获取集合统计信息
```

**Task 2.1.2: RabbitMQ客户端**

```python
# 文件: python-services/shared/rabbit_client.py

class RabbitMQClient:
    """RabbitMQ消息队列客户端"""
    
    async def connect(self):
        # 建立连接
        
    async def publish_document_processed(self, message: Dict):
        # 发布文档处理完成消息
        
    async def publish_vectorization_completed(self, message: Dict):
        # 发布向量化完成消息
        
    async def setup_queues(self):
        # 创建必要的队列和交换机
```

**Task 2.1.3: 配置管理**

```python
# 文件: python-services/shared/config.py

from pydantic import BaseSettings

class Settings(BaseSettings):
    # Milvus配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "document_store"
    
    # RabbitMQ配置
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672"
    
    # 向量化模型配置
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # Ollama配置
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama2"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

**验收标准 2.1:**

- [ ] Milvus客户端连接成功
- [ ] 基本CRUD操作正常 (插入、查询、删除)
- [ ] RabbitMQ连接和消息发布成功
- [ ] 配置管理加载环境变量正确
- [ ] 共享组件可被其他服务导入

---

### 子阶段2.2：文档处理服务 (4-5天) 📄

**状态**: 🔴 待开始  
**优先级**: 🔥 高  
**依赖**: 子阶段2.1完成  
**并行**: 可与2.3并行开发  

#### 🎯 具体任务清单

**Task 2.2.1: 现有代码迁移**

```bash
# 迁移映射
src/data_processing/processors/     → python-services/document-processor/src/processors/
src/data_processing/splitting/      → python-services/document-processor/src/splitting/
src/utils/logging_utils.py         → python-services/shared/utils/
```

**Task 2.2.2: FastAPI服务框架**

```python
# 文件: python-services/document-processor/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List
import asyncio

app = FastAPI(title="Document Processor Service", version="1.0.0")

class DocumentProcessRequest(BaseModel):
    document_id: str
    filename: str

class ProcessingResult(BaseModel):
    document_id: str
    chunks: List[Dict]
    status: str
    processing_time: float

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "document-processor"}

@app.post("/process-document", response_model=ProcessingResult)
async def process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_id: str = None
):
    """处理上传的文档，返回分块结果"""
    # 实现文档处理逻辑

@app.get("/processing-status/{document_id}")
async def get_processing_status(document_id: str):
    """获取文档处理状态"""
    # 实现状态查询

@app.post("/reprocess-document/{document_id}")
async def reprocess_document(document_id: str):
    """重新处理文档"""
    # 实现重新处理逻辑
```

**Task 2.2.3: 文档处理器核心逻辑**

```python
# 文件: python-services/document-processor/src/document_processor.py

class DocumentProcessor:
    """文档处理器主类，整合现有处理逻辑"""
    
    def __init__(self):
        self.rabbit_client = RabbitMQClient()
        # 初始化各种处理器
        
    async def process_file(self, 
                          file_content: bytes, 
                          filename: str, 
                          mime_type: str,
                          document_id: str) -> List[DocumentChunk]:
        """
        处理文件的主入口
        1. 根据文件类型选择处理器
        2. 提取文本内容
        3. 分块处理
        4. 发送到消息队列
        """
        
    def _select_processor(self, mime_type: str):
        """根据MIME类型选择处理器"""
        
    def _chunk_text(self, text: str, metadata: Dict) -> List[DocumentChunk]:
        """文本分块，复用现有splitting逻辑"""
        
    async def _notify_processing_complete(self, result: ProcessingResult):
        """发送处理完成通知"""
```

**验收标准 2.2:**

- [ ] 支持PDF、Word、TXT、Markdown文档处理
- [ ] 文档分块功能正常，块大小合理
- [ ] 处理结果正确发送到RabbitMQ
- [ ] API响应时间 < 10秒 (小文档)
- [ ] 错误处理和状态反馈完善

---

### 子阶段2.3：向量化服务 (5-6天) 🧠

**状态**: 🔴 待开始  
**优先级**: ⚡ 中高  
**依赖**: 子阶段2.1完成  
**并行**: 可与2.2并行开发  

#### 🎯 具体任务清单

**Task 2.3.1: 向量化模型选择和优化**

```python
# 文件: python-services/vector-service/src/vectorizer.py

class CPUVectorizer:
    """CPU优化的向量化器"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """延迟加载模型"""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            
    def vectorize_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """批量向量化，优化性能"""
        
    def vectorize_single(self, text: str) -> List[float]:
        """单文本向量化"""
        
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "embedding_dim": 384,  # all-MiniLM-L6-v2
            "max_seq_length": 256
        }
```

**Task 2.3.2: FastAPI向量化服务**

```python
# 文件: python-services/vector-service/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Vector Service", version="1.0.0")

class VectorizeRequest(BaseModel):
    texts: List[str]
    document_id: Optional[str] = None
    store_in_milvus: bool = True

class VectorizeResponse(BaseModel):
    vectors: Optional[List[List[float]]] = None
    vector_ids: Optional[List[str]] = None
    model_info: Dict
    processing_time: float

@app.get("/health")
async def health_check():
    vectorizer = CPUVectorizer()
    return {
        "status": "healthy", 
        "service": "vector-service",
        "model_info": vectorizer.get_model_info()
    }

@app.post("/vectorize", response_model=VectorizeResponse)
async def vectorize_texts(request: VectorizeRequest):
    """向量化文本列表"""
    
@app.post("/vectorize-and-store")
async def vectorize_and_store(request: VectorizeRequest):
    """向量化并存储到Milvus"""
    
@app.get("/search")
async def search_similar_texts(query: str, top_k: int = 5):
    """语义搜索接口"""
```

**Task 2.3.3: Milvus集成优化**

```python
# 文件: python-services/vector-service/src/vector_manager.py

class VectorManager:
    """向量管理器，专门处理向量存储和检索"""
    
    def __init__(self):
        self.milvus_client = MilvusClient()
        self.vectorizer = CPUVectorizer()
        
    async def store_vectors(self, 
                          texts: List[str], 
                          document_id: str,
                          metadata: List[Dict] = None) -> List[str]:
        """存储向量到Milvus"""
        
    async def search_similar(self, 
                           query_text: str, 
                           top_k: int = 5,
                           filters: Dict = None) -> List[Dict]:
        """语义相似搜索"""
        
    async def get_vectors_by_document(self, document_id: str) -> List[Dict]:
        """根据文档ID获取向量"""
        
    async def delete_document_vectors(self, document_id: str) -> bool:
        """删除文档的所有向量"""
```

**验收标准 2.3:**

- [ ] 向量化性能 < 2秒/100文本
- [ ] Milvus存储和检索功能正常
- [ ] 支持批量处理和单文本处理
- [ ] 语义搜索准确性可接受
- [ ] 支持并发请求 (10+ concurrent)

---

### 子阶段2.4：RAG检索服务 (4-5天) 🔍

**状态**: 🔴 待开始  
**优先级**: ⚡ 中高  
**依赖**: 子阶段2.3完成  

#### 🎯 具体任务清单

**Task 2.4.1: 现有RAG代码迁移**

```bash
# 迁移映射
src/rag/retriever.py               → python-services/rag-service/src/retriever.py
src/chains/processors/             → python-services/rag-service/src/processors/
src/model/ollama_client.py         → python-services/rag-service/src/llm_client.py
```

**Task 2.4.2: RAG检索服务API**

```python
# 文件: python-services/rag-service/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict

app = FastAPI(title="RAG Service", version="1.0.0")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    include_sources: bool = True
    document_filters: Optional[Dict] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    query_time: float
    confidence_score: Optional[float] = None

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "rag-service"}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """处理RAG查询请求"""
    
@app.post("/chat")
async def chat_completion(messages: List[Dict]):
    """多轮对话接口"""
    
@app.get("/documents/{document_id}/qa")
async def document_qa(document_id: str, question: str):
    """针对特定文档的问答"""
```

**Task 2.4.3: RAG处理器核心逻辑**

```python
# 文件: python-services/rag-service/src/rag_processor.py

class RAGProcessor:
    """RAG处理器，整合检索和生成"""
    
    def __init__(self):
        self.retriever = DocumentRetriever()
        self.llm_client = OllamaClient()
        self.vector_service_url = settings.vector_service_url
        
    async def process_query(self, 
                          query: str, 
                          top_k: int = 5,
                          filters: Dict = None) -> QueryResponse:
        """
        RAG查询处理主流程:
        1. 向量化查询
        2. 检索相关文档
        3. 构建提示词
        4. 生成回答
        5. 格式化结果
        """
        
    async def _vectorize_query(self, query: str) -> List[float]:
        """调用向量化服务"""
        
    async def _retrieve_documents(self, 
                                query_vector: List[float], 
                                top_k: int,
                                filters: Dict = None) -> List[Dict]:
        """从Milvus检索相关文档"""
        
    def _build_prompt(self, query: str, documents: List[Dict]) -> str:
        """构建LLM提示词"""
        
    async def _generate_answer(self, prompt: str) -> str:
        """调用LLM生成答案"""
```

**Task 2.4.4: LLM客户端优化**

```python
# 文件: python-services/rag-service/src/llm_client.py

class OllamaClient:
    """Ollama LLM客户端，基于现有ollama_client.py"""
    
    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model
        
    async def generate_answer(self, 
                            prompt: str, 
                            max_tokens: int = 500,
                            temperature: float = 0.1) -> str:
        """生成答案"""
        
    async def chat_completion(self, messages: List[Dict]) -> str:
        """多轮对话"""
        
    async def check_health(self) -> bool:
        """检查Ollama服务状态"""
```

**验收标准 2.4:**

- [ ] RAG查询准确性满足要求
- [ ] 响应时间 < 3秒
- [ ] 支持复杂查询和多轮对话
- [ ] 错误处理完善
- [ ] LLM集成稳定

---

### 子阶段2.5：集成测试和优化 (3-4天) 🧪

**状态**: 🔴 待开始  
**优先级**: ⚡ 中等  
**依赖**: 所有前置子阶段完成  

#### 🎯 具体任务清单

**Task 2.5.1: 端到端测试脚本**

```python
# 文件: python-services/tests/test_e2e.py

class TestE2EFlow:
    """端到端流程测试"""
    
    async def test_document_upload_to_query(self):
        """完整流程: 上传→处理→向量化→查询"""
        # 1. 上传测试文档
        # 2. 等待处理完成
        # 3. 验证向量存储
        # 4. 执行查询测试
        # 5. 验证结果质量
        
    async def test_concurrent_processing(self):
        """并发处理测试"""
        
    async def test_large_document_handling(self):
        """大文档处理测试"""
        
    async def test_error_scenarios(self):
        """异常场景测试"""
```

**Task 2.5.2: 性能基准测试**

```python
# 文件: python-services/tests/test_performance.py

class PerformanceTests:
    """性能基准测试"""
    
    def test_vectorization_performance(self):
        """向量化性能测试"""
        # 目标: 100文本 < 2秒
        
    def test_query_response_time(self):
        """查询响应时间测试"""
        # 目标: RAG查询 < 3秒
        
    def test_concurrent_load(self):
        """并发负载测试"""
        # 目标: 支持10+并发
```

**Task 2.5.3: 系统优化**

```python
# 优化清单
- [ ] 内存使用优化 (模型加载策略)
- [ ] 数据库连接池优化
- [ ] 缓存策略实现
- [ ] 错误处理完善
- [ ] 日志记录标准化
- [ ] 监控指标收集
```

**验收标准 2.5:**

- [ ] 完整端到端流程无错误
- [ ] 性能指标达标
- [ ] 并发处理稳定
- [ ] 错误处理完善
- [ ] 具备生产部署条件

---

## 📊 执行进度跟踪

### 进度检查点

- [ ] **Day 4**: 子阶段2.1完成，基础设施就绪
- [ ] **Day 9**: 子阶段2.2+2.3完成，文档处理和向量化服务就绪
- [ ] **Day 14**: 子阶段2.4完成，RAG服务就绪
- [ ] **Day 17**: 子阶段2.5完成，整体测试通过

### 风险缓解计划

1. **模型性能风险**: 预备多个模型选项，优先轻量级模型
2. **集成复杂度风险**: 每个子阶段独立测试验证
3. **时间延期风险**: 采用MVP原则，核心功能优先

---

## 🔧 技术规范

### API设计规范

```yaml
# 统一API响应格式
{
  "status": "success|error",
  "data": {},
  "message": "string",
  "timestamp": "ISO8601",
  "execution_time": "float"
}

# 错误响应格式
{
  "status": "error",
  "error_code": "string", 
  "message": "string",
  "details": {}
}
```

### 日志规范

```python
# 统一日志格式
{
  "timestamp": "2024-01-01T00:00:00Z",
  "level": "INFO|ERROR|DEBUG",
  "service": "document-processor|vector-service|rag-service",
  "message": "string",
  "request_id": "uuid",
  "execution_time": "float",
  "metadata": {}
}
```

### 测试覆盖率要求

- 单元测试覆盖率 > 80%
- API接口测试覆盖率 100%
- 端到端测试覆盖主要流程

---

## 📚 参考资料和依赖

### 现有代码引用路径

```bash
# 需要迁移的核心文件
src/data_processing/processors/pdf_processor.py
src/data_processing/processors/docx_processor.py
src/data_processing/processors/txt_processor.py
src/data_processing/splitting/text_splitter.py
src/data_processing/vectorization/base.py
src/data_processing/storage/milvus_store.py
src/rag/retriever.py
src/chains/processors/query_processor.py
src/model/ollama_client.py
src/utils/logging_utils.py
```

### 关键配置参数

```yaml
# 向量化配置
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
embedding_dimension: 384
batch_size: 32
max_seq_length: 256

# Milvus配置
collection_name: "document_store"
metric_type: "COSINE"
index_type: "IVF_FLAT"
nlist: 128

# 性能目标
vectorization_speed: "< 2s per 100 texts"
query_response_time: "< 3s"
concurrent_requests: "> 10"
```

---

## ✅ 最终交付清单

### 代码交付

- [ ] 3个完整的FastAPI微服务
- [ ] 共享组件库
- [ ] 完整的测试套件
- [ ] Docker配置文件
- [ ] API文档

### 文档交付  

- [ ] API接口文档
- [ ] 部署运维指南
- [ ] 测试报告
- [ ] 性能基准报告

### 验证交付

- [ ] 完整端到端流程演示
- [ ] 性能测试报告
- [ ] 错误处理验证
- [ ] 并发处理验证

---

**执行原则**:

1. **MVP优先** - 核心功能先实现，优化后续进行
2. **测试驱动** - 每个功能模块都要有对应测试
3. **文档同步** - 代码和文档同步更新
4. **持续验证** - 每日进度检查和问题解决

此文档将作为阶段2执行的最高指导，所有任务都应该按照此文档的规范和标准进行。
