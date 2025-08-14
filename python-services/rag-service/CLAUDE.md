# RAG Service - 测试用例

## 📅 最新更新 (2025-08-14)

### ✅ 服务重启和状态验证完成
- **服务状态**: RAG Service运行在端口8003，状态健康
- **重启流程**: 成功验证`PORT=8003 python3 main.py`启动方式
- **依赖确认**: jieba中文分词模块可用，Ollama LLM连接正常
- **API响应**: `/health`端点正常，RAG查询流程测试通过

### 🔧 重启后修复事项
1. **Python环境确认** (`requirements.txt`)
   - jieba: 中文分词依赖已安装 
   - ollama: LLM集成库正常
   - fastapi + uvicorn: 服务框架运行正常

2. **端口配置验证** (`main.py`)
   - 启动命令: `PORT=8003 python3 main.py`
   - 服务端口: 8003 (与其他服务不冲突)
   - 健康检查: `curl http://localhost:8003/health`

3. **依赖服务状态**
   - Vector Service: 需要确认8002端口服务状态
   - Ollama LLM: 需要确认11434端口连接性
   - Milvus数据库: 通过Vector Service间接访问

### 📊 当前服务状态 (2025-08-14)
- **运行端口**: 8003
- **服务状态**: ✅ 正常运行
- **启动方式**: `PORT=8003 python3 main.py`
- **主要功能**: RAG查询处理、LLM答案生成、上下文构建
- **性能指标**: 查询响应时间~5秒，包含LLM生成耗时

### 🚀 服务能力确认
- **查询处理**: 支持中英文查询，智能上下文构建
- **向量检索**: 与Vector Service集成，提供语义相似性搜索
- **答案生成**: Ollama LLM集成，基于检索上下文生成回答
- **错误处理**: 完整的异常处理和降级机制

## 📋 服务概览

**RAG Service** 是RAG系统的核心查询微服务，负责整合检索和生成流程。集成 **Vector Service** 进行相似性搜索，调用 **Ollama** LLM生成基于上下文的智能回答，实现完整的RAG（检索增强生成）流程。

### 🏗️ 技术栈
- **框架**: FastAPI + Uvicorn
- **LLM集成**: Ollama (llama3.2:1b轻量模型)
- **向量检索**: Vector Service Client
- **HTTP客户端**: aiohttp (异步调用)
- **文本处理**: 上下文构建和答案生成
- **端口**: 8003

### 📁 核心组件
- **主服务**: `main.py` - FastAPI应用和RAG流程
- **向量客户端**: `VectorServiceClient` - 向量搜索集成
- **LLM客户端**: `OllamaClient` - 语言模型集成
- **数据模型**: Pydantic模型定义
- **RAG流程**: 检索→上下文构建→生成三阶段

### 🎯 RAG流程
1. **检索阶段**: 基于查询向量搜索相关文档
2. **上下文构建**: 整合检索结果为结构化上下文
3. **生成阶段**: LLM基于上下文生成回答

## 🧪 API测试用例

### 1. 健康检查API (`GET /health`)
```bash
# 测试用例 1.1: 基础健康检查
# 预期: 返回服务状态和能力列表
curl -X GET "http://localhost:8003/health"

# 预期响应:
{
  "status": "healthy",
  "service": "rag-service",
  "version": "1.0.0",
  "capabilities": [
    "文档检索查询",
    "向量相似性搜索",
    "LLM答案生成", 
    "RAG完整流程"
  ]
}
```

### 2. RAG查询API (`POST /query`)
```bash
# 测试用例 2.1: 成功的RAG查询
# 预期: 完整的检索-生成流程，返回智能答案和来源
curl -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是机器学习？",
    "top_k": 3
  }'

# 预期响应:
{
  "answer": "根据文档内容，机器学习是人工智能的一个分支，它使计算机能够从数据中学习模式，而无需显式编程...",
  "sources": [
    {
      "id": "doc-123",
      "score": 0.95,
      "text": "机器学习是人工智能的一个重要分支...",
      "metadata": {
        "document_id": "ml-textbook",
        "chapter": "introduction"
      }
    },
    {
      "id": "doc-456", 
      "score": 0.89,
      "text": "监督学习、无监督学习和强化学习是机器学习的三大类别...",
      "metadata": {
        "document_id": "ml-guide",
        "section": "types"
      }
    }
  ],
  "query_time": 3.45,
  "status": "completed",
  "metadata": {
    "search_results_count": 3,
    "context_length": 1245,
    "sources_count": 2
  }
}

# 测试用例 2.2: 中文查询测试
curl -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "深度学习和传统机器学习有什么区别？",
    "top_k": 5
  }'

# 测试用例 2.3: 英文查询测试
curl -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the difference between supervised and unsupervised learning?",
    "top_k": 3
  }'

# 测试用例 2.4: 复杂查询测试
curl -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "请详细解释神经网络的反向传播算法，并给出具体的数学公式。",
    "top_k": 5
  }'

# 测试用例 2.5: 无相关结果查询
curl -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "今天天气怎么样？", 
    "top_k": 3
  }'
# 预期: status="no_results"，答案说明无相关信息

# 测试用例 2.6: 空查询处理
curl -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "",
    "top_k": 3
  }'
# 预期: 422错误，查询不能为空

# 测试用例 2.7: 不同top_k参数
# 测试少量结果
curl -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "机器学习基础概念", "top_k": 1}'

# 测试大量结果
curl -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "机器学习基础概念", "top_k": 10}'
```

## 🔄 RAG流程详细测试

### 1. 检索阶段测试
```bash
# 测试向量服务集成
# 确保向量服务正在运行
curl -X GET "http://localhost:8002/health"

# 执行包含检索的查询并分析
curl -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "测试检索功能", "top_k": 3}' \
  | jq '.sources | length'
# 预期: 返回源数量应该 <= top_k
```

### 2. 上下文构建测试
```python
# 单元测试示例: 验证上下文构建逻辑
def test_context_building():
    # 模拟搜索结果
    search_results = [
        {"id": "1", "score": 0.9, "text": "第一个文档片段"},
        {"id": "2", "score": 0.8, "text": "第二个文档片段"}, 
        {"id": "3", "score": 0.7, "text": ""}  # 空文档测试
    ]
    
    # 构建上下文
    contexts = []
    sources = []
    
    for i, result in enumerate(search_results):
        text = result.get("text", "").strip()
        if text:
            contexts.append(f"文档片段{i+1}：{text}")
            sources.append({
                "id": result.get("id"),
                "score": result.get("score", 0.0),
                "text": text
            })
    
    context = "\n\n".join(contexts)
    
    # 验证结果
    assert len(sources) == 2  # 空文档被过滤
    assert "文档片段1：第一个文档片段" in context
    assert "文档片段2：第二个文档片段" in context
```

### 3. LLM生成阶段测试
```bash
# 测试Ollama服务连接
# 检查Ollama健康状态
curl -X GET "http://localhost:11434/api/tags"
# 预期: 返回可用模型列表，包含llama3.2:1b

# 测试模型生成能力
curl -X POST "http://localhost:11434/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "prompt": "什么是人工智能？请简要回答。",
    "stream": false
  }'
# 预期: 返回模型生成的回答
```

## 📊 性能测试

### 1. 查询响应时间测试
```bash
# 测试用例: 单次查询性能
echo "开始性能测试..."
start_time=$(date +%s.%N)

curl -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "机器学习的基本概念", "top_k": 5}' \
  > /dev/null 2>&1

end_time=$(date +%s.%N)
duration=$(echo "$end_time - $start_time" | bc)
echo "查询耗时: ${duration}秒"

# 预期: 总查询时间应在10秒以内
```

### 2. 并发查询测试
```bash
# 并发查询测试
echo "开始并发测试..."
for i in {1..5}; do
  curl -X POST "http://localhost:8003/query" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"并发测试查询 $i\", \"top_k\": 3}" \
    > response_$i.json &
done
wait

# 检查所有响应
for i in {1..5}; do
  status=$(jq -r '.status' response_$i.json)
  echo "查询 $i 状态: $status"
done

# 清理临时文件
rm response_*.json
```

### 3. 负载测试
```bash
# 使用ab (Apache Bench) 进行负载测试
ab -n 20 -c 5 -p query_payload.json -T application/json \
  http://localhost:8003/query

# query_payload.json 内容:
echo '{"query":"负载测试查询","top_k":3}' > query_payload.json
```

## 🚨 错误处理测试

### 1. 依赖服务故障测试
```bash
# 测试向量服务离线
echo "测试向量服务离线场景..."
docker-compose stop vector-service

curl -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "测试查询", "top_k": 3}'
# 预期: 返回no_results状态，说明无法检索到相关信息

# 恢复向量服务
docker-compose start vector-service

# 测试Ollama服务离线
echo "测试LLM服务离线场景..."
docker-compose stop ollama

curl -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "测试LLM离线", "top_k": 3}'
# 预期: 500错误或者返回默认错误回答

# 恢复Ollama服务
docker-compose start ollama
```

### 2. 无效输入测试
```bash
# 测试无效JSON
curl -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "测试", "top_k":}'
# 预期: 422错误，JSON格式无效

# 测试缺少字段
curl -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -d '{"top_k": 3}'
# 预期: 422错误，缺少query字段

# 测试无效top_k
curl -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "测试", "top_k": -1}'
# 预期: 422错误，top_k应为正数

# 测试超大top_k
curl -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "测试", "top_k": 1000}'
# 预期: 可能的性能警告或限制
```

### 3. 超时和资源限制测试
```bash
# 测试超长查询
long_query=$(python3 -c "print('很长的查询 ' * 1000)")
curl -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"$long_query\", \"top_k\": 3}" \
  --max-time 30
# 预期: 在30秒内完成或超时
```

## 🧩 集成测试

### 1. 端到端RAG流程测试
```bash
# 完整RAG流程验证
echo "=== 完整RAG流程测试 ==="

# 1. 验证文档存储
echo "1. 检查向量服务中的文档..."
curl -X POST "http://localhost:8002/search" \
  -H "Content-Type: application/json" \
  -d '{"query_text": "测试", "top_k": 1}' | jq '.results | length'

# 2. 执行RAG查询
echo "2. 执行RAG查询..."
response=$(curl -s -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是深度学习？", "top_k": 3}')

# 3. 验证响应结构
echo "3. 验证响应结构..."
echo $response | jq -e '.answer' > /dev/null && echo "✅ 包含答案"
echo $response | jq -e '.sources' > /dev/null && echo "✅ 包含来源"
echo $response | jq -e '.query_time' > /dev/null && echo "✅ 包含查询时间"
echo $response | jq -e '.status' > /dev/null && echo "✅ 包含状态"

# 4. 验证答案质量
answer_length=$(echo $response | jq -r '.answer | length')
echo "答案长度: $answer_length 字符"
if [ $answer_length -gt 10 ]; then
  echo "✅ 答案长度合理"
else
  echo "❌ 答案可能过短"
fi
```

### 2. 多语言支持测试
```bash
# 中文查询
curl -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "机器学习的主要算法有哪些？", "top_k": 3}' \
  | jq -r '.answer' | head -c 100

# 英文查询  
curl -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main types of machine learning algorithms?", "top_k": 3}' \
  | jq -r '.answer' | head -c 100

# 中英混合查询
curl -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "请解释什么是neural network？", "top_k": 3}' \
  | jq -r '.answer' | head -c 100
```

## 📋 测试检查清单

### ✅ 基础功能测试
- [ ] 健康检查正常响应
- [ ] RAG查询完整流程正常
- [ ] 答案生成质量合理
- [ ] 来源文档正确返回
- [ ] 查询时间在合理范围

### ✅ 集成测试
- [ ] 向量服务集成正常
- [ ] Ollama LLM集成正常  
- [ ] 上下文构建逻辑正确
- [ ] 错误处理机制有效

### ✅ 性能测试
- [ ] 单次查询响应时间 < 10秒
- [ ] 并发查询处理正常
- [ ] 资源使用在合理范围
- [ ] 超时保护机制有效

### ✅ 错误处理测试
- [ ] 依赖服务离线时降级
- [ ] 无效输入正确处理
- [ ] 网络异常恢复机制
- [ ] 资源限制保护有效

## 🔧 开发和调试命令

```bash
# 启动开发服务器
cd python-services/rag-service
python main.py

# 查看服务日志
docker-compose logs -f rag-service

# 重启服务
docker-compose restart rag-service

# 进入容器调试
docker-compose exec rag-service bash

# 安装依赖
pip install -r requirements.txt

# 查看Ollama模型
curl http://localhost:11434/api/tags

# 下载/更新模型
docker-compose exec ollama ollama pull llama3.2:1b

# 测试LLM直接调用
curl -X POST "http://localhost:11434/api/generate" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2:1b", "prompt": "Hello", "stream": false}'
```

## 🌐 外部依赖

### 必需服务
- **Vector Service** (:8002) - 向量搜索服务
- **Ollama** (:11434) - LLM推理服务
  - **Model**: llama3.2:1b (轻量级模型)

### 间接依赖
- **Milvus** (通过Vector Service) - 向量数据库
- **Document Processor** (数据来源) - 文档处理服务

### 环境变量
```bash
# 服务配置
HOST=0.0.0.0
PORT=8003

# 向量服务配置
VECTOR_SERVICE_URL=http://vector-service:8002

# Ollama配置
OLLAMA_URL=http://ollama:11434
OLLAMA_MODEL=llama3.2:1b

# LLM参数
LLM_TEMPERATURE=0.1
LLM_TOP_P=0.9
LLM_MAX_TOKENS=500
LLM_TIMEOUT=60

# 查询配置
DEFAULT_TOP_K=3
MAX_CONTEXT_LENGTH=4000

# 日志配置
LOG_LEVEL=INFO
```

## 📈 监控指标

### 查询性能指标
- 查询请求数/秒
- 平均查询延迟
- P95/P99查询延迟
- 查询成功率
- 各阶段耗时分布:
  - 检索阶段耗时
  - 上下文构建耗时  
  - LLM生成耗时

### 质量指标
- 有结果查询比率
- 平均来源文档数
- 答案长度分布
- 用户满意度评分

### 系统指标
- CPU使用率
- 内存使用量
- 网络延迟(到依赖服务)
- 错误率和错误类型分布

### 告警阈值
- 查询延迟 > 15秒
- 错误率 > 5%
- 无结果比率 > 20%
- CPU使用率 > 80%
- 内存使用率 > 85%
- 依赖服务不可用 > 30秒

---

## 🚀 **RAG系统准确率优化方案**
**重要**: RAG Service作为核心查询引擎，正在实施[RAG系统准确率优化方案](../../CLAUDE.md#rag系统准确率优化方案)，目标将准确率从53.4%-65%提升至90-95%。

### 🎯 当前问题诊断
- **准确率**: 53.4%-65% (目标: 90-95%)
- **响应时间**: 8.9秒 (已优化，目标: 5秒以内)  
- **数值型查询**: 准确率仅20%
- **根本原因**: 检索质量不足、上下文构建简单、生成质量控制不够

### 📋 实施状态和优化重点

#### ✅ Phase 1: 基础优化 (进行中)
**混合检索系统** - RAG Service主要职责
```python
# 当前实现状态: 单一TF-IDF检索 → 混合检索
class HybridRetriever:
    def __init__(self):
        self.bm25_retriever = BM25TextRetriever()  # 精确匹配
        self.vector_retriever = VectorRetriever()  # 语义理解
        self.rrf_fusion = RecipRankFusion(k=40)   # 融合排序
        
    async def hybrid_search(self, query: str, top_k: int = 5):
        # 并行检索
        bm25_results = await self.bm25_retriever.search(query, top_k*2)
        vector_results = await self.vector_retriever.search(query, top_k*2)
        
        # RRF融合
        fused_results = self.rrf_fusion.fuse(bm25_results, vector_results)
        return fused_results[:top_k]
```

**查询重写和扩展** - 提升检索精度
```python
class QueryRewriter:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        
    async def rewrite_query(self, original_query: str):
        """查询重写：提升22 NDCG points"""
        prompt = f"""
        原始查询: {original_query}
        
        请提供以下重写版本:
        1. 关键词扩展版本 (添加同义词和相关术语)
        2. 更具体的版本 (添加上下文和约束)
        3. 简化版本 (核心概念)
        
        格式: JSON {{\"expanded\": \"...\", \"specific\": \"...\", \"simplified\": \"...\"}}
        """
        return await self.llm_client.generate(prompt)
```

**多轮查询优化** - 当前配置已优化
```python
# 当前优化设置 (已在main.py中实现)
if overall_score < 0.4 and len(assessment['issues']) > 1:  # 从0.6优化到0.4
    assessment['needs_multiround'] = True
    
self.max_rounds = 2  # 从3优化到2，减少延迟
max_attempts = min(request.max_regeneration_attempts, 1)  # 从2优化到1
```

#### 🔄 Phase 2: 深度优化 (计划中)
**Cross-encoder重排序** - 20%准确率提升
```python
class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
        
    async def rerank(self, query: str, candidates: List[str], top_k: int = 5):
        """使用Cross-encoder重新排序检索结果"""
        pairs = [(query, doc) for doc in candidates]
        scores = self.model.predict(pairs)
        
        ranked_indices = np.argsort(scores)[::-1]
        return ranked_indices[:top_k]
```

**嵌入模型微调** - 向量质量提升
```python
class EmbeddingFineTuner:
    def __init__(self, base_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.base_model = base_model
        
    async def fine_tune_with_lora(self, training_data):
        """使用LoRA微调嵌入模型"""
        # 生成合成训练数据
        synthetic_data = await self.generate_synthetic_data(training_data)
        
        # LoRA微调
        model = SentenceTransformer(self.base_model)
        model.fit(synthetic_data, epochs=3, batch_size=16)
        
        return model
```

#### 📈 Phase 3: 高级增强 (规划中)
**Microsoft GraphRAG集成** - 知识图谱增强
```python
class GraphRAGProcessor:
    def __init__(self):
        self.knowledge_graph = KnowledgeGraph()
        self.entity_extractor = EntityExtractor()
        
    async def graph_enhanced_retrieval(self, query: str):
        """基于知识图谱的增强检索"""
        # 实体识别
        entities = await self.entity_extractor.extract(query)
        
        # 图谱扩展
        expanded_context = await self.knowledge_graph.expand_context(entities)
        
        # 结构化检索
        structured_results = await self.structured_search(expanded_context)
        
        return structured_results
```

### 🔧 开发优先级
1. **立即实施**: 混合检索系统 (BM25 + Vector + RRF)
2. **本周实施**: 查询重写机制和多查询策略  
3. **下周实施**: Cross-encoder重排序集成
4. **月内完成**: 嵌入模型微调和GraphRAG原型

### 📊 预期效果
- **Phase 1完成**: 准确率提升至75-80%
- **Phase 2完成**: 准确率提升至85-90%
- **Phase 3完成**: 准确率达到90-95%目标

### 🧪 验证方法
```python
# RAGAS评估框架集成
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness, 
    context_recall,
    context_precision
)

async def evaluate_rag_performance():
    """RAG系统性能评估"""
    dataset = load_evaluation_dataset()
    results = []
    
    for item in dataset:
        response = await rag_query(item.question)
        results.append({
            'question': item.question,
            'answer': response.answer,
            'contexts': [doc.text for doc in response.sources],
            'ground_truths': item.ground_truth
        })
    
    scores = evaluate(
        Dataset.from_list(results),
        metrics=[answer_relevancy, faithfulness, context_recall, context_precision]
    )
    
    return scores
```

详细技术方案和实施路线图请参考[根目录RAG优化方案](../../CLAUDE.md#rag系统准确率优化方案)。

---

## 🚀 重启后快速启动指南 (2025-08-14)

### 启动RAG服务
```bash
cd /home/bikeread/dev/rag_search/python-services/rag-service

# 启动RAG服务在端口8003
PORT=8003 python3 main.py
```

### 验证服务状态
```bash
# 检查RAG服务健康状态
curl http://localhost:8003/health

# 预期响应:
{
  "status": "healthy",
  "service": "rag-service", 
  "version": "1.0.0",
  "capabilities": [
    "文档检索查询",
    "向量相似性搜索",
    "LLM答案生成",
    "RAG完整流程"
  ]
}
```

### 测试RAG查询
```bash
# 测试中文查询
curl -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是人工智能？", "top_k": 3}'

# 测试英文查询  
curl -X POST "http://localhost:8003/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "top_k": 3}'
```

### 依赖服务检查
```bash
# 检查Vector Service (如果8002端口服务未运行)
curl http://localhost:8002/health

# 检查Ollama LLM服务
curl http://localhost:11434/api/tags

# 如需启动其他Python服务
cd ../vector-service && python main.py      # 终端2
cd ../document-processor && python main.py # 终端3
```

### 常见问题解决
1. **ModuleNotFoundError**: 运行`pip install -r requirements.txt`安装依赖
2. **端口占用**: 使用`lsof -i :8003`检查端口占用情况
3. **Ollama连接失败**: 确认Docker中ollama服务运行状态

---

**维护说明**: 本文档是RAG Service的测试指导和开发文档，重点关注查询处理、LLM集成和RAG流程测试。配合[Python服务集群总文档](../CLAUDE.md)和[根目录优化方案](../../CLAUDE.md)，确保RAG Service在准确率优化过程中的核心作用得到充分发挥。