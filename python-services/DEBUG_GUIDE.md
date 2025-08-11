# Python服务本地调试指南

## 🚀 快速开始

### 1. 停止Docker中的Python服务
```bash
docker stop rag_search-document-processor-1 rag_search-vector-service-1 rag_search-rag-service-1
```

### 2. 确保依赖服务运行
```bash
# 保持这些服务在Docker中运行
docker-compose up -d postgres redis rabbitmq milvus ollama
```

### 3. 启动本地Python服务

#### 方式一：使用启动脚本（推荐）
```bash
cd python-services
./start_local.sh
```

#### 方式二：手动启动单个服务
```bash
cd python-services

# 激活虚拟环境
source venv/bin/activate

# 设置环境变量
export $(cat .env.local | grep -v '^#' | xargs)

# 启动特定服务
cd document-processor && python main.py  # 终端1
cd vector-service && python main.py      # 终端2
cd rag-service && python main.py         # 终端3
```

#### 方式三：VS Code调试
1. 打开VS Code
2. 打开 `/python-services` 目录
3. 选择调试配置：
   - `Document Processor` - 调试文档处理服务
   - `Vector Service` - 调试向量服务
   - `RAG Service` - 调试RAG服务
   - `All Services (Parallel)` - 同时调试所有服务
4. 按F5启动调试

## 🔍 调试技巧

### 1. 添加断点
在需要调试的代码行左侧点击，添加红色断点。

### 2. 查看日志
```bash
# 实时查看服务日志
tail -f document-processor.log
tail -f vector-service.log
tail -f rag-service.log
```

### 3. 测试API
```bash
# 健康检查
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health

# 文档处理测试
curl -X POST http://localhost:8001/process-document \
  -F "file=@test.pdf" \
  -F "document_id=test-001"

# 向量搜索测试
curl -X POST http://localhost:8002/search \
  -H "Content-Type: application/json" \
  -d '{"query_text": "测试查询", "top_k": 3}'

# RAG查询测试
curl -X POST http://localhost:8003/query \
  -H "Content-Type: application/json" \
  -d '{"query": "测试问题", "top_k": 3}'
```

## 🛠️ 常见问题

### 1. 端口被占用
```bash
# 查看端口占用
lsof -i:8001
lsof -i:8002
lsof -i:8003

# 杀死进程
kill -9 <PID>
```

### 2. Milvus连接失败
```bash
# 检查Milvus是否运行
docker ps | grep milvus

# 重启Milvus
docker restart rag_search-milvus-standalone-1
```

### 3. Python依赖问题
```bash
# 重新安装依赖
pip install -r shared/requirements.txt --force-reinstall
```

### 4. 环境变量未加载
```bash
# 手动加载环境变量
source .env.local
# 或
export $(cat .env.local | grep -v '^#' | xargs)
```

## 📝 调试重点区域

### Document Processor
- `process_document()` - 文档处理主逻辑
- `extract_text()` - 文本提取
- `chunk_text()` - 文本分块

### Vector Service  
- `vectorize_texts()` - 文本向量化
- `search_vectors()` - 向量搜索
- `CPUVectorizer` - TF-IDF向量化器

### RAG Service
- `process_rag_query()` - RAG查询主流程
- `build_context()` - 上下文构建
- `generate_answer()` - 答案生成

## 🔄 切换回Docker模式

```bash
# 停止本地服务
pkill -f 'python main.py'

# 重启Docker服务
docker-compose up -d document-processor vector-service rag-service
```

## 💡 性能优化建议

1. **向量化优化**
   - 调整TF-IDF参数
   - 尝试不同的向量维度
   - 实现批处理

2. **搜索优化**
   - 调整Milvus索引参数
   - 实现结果缓存
   - 优化重排序算法

3. **LLM优化**
   - 调整温度和top_k参数
   - 优化prompt模板
   - 实现流式响应

---

**提示**: 本地调试时LOG_LEVEL设置为DEBUG，会输出详细日志，方便追踪问题。