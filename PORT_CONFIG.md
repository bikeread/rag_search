# RAG系统端口分配规范

## 📋 统一端口分配表

| 服务类型 | 端口 | 服务名称 | 协议 | 描述 |
|---------|------|----------|------|------|
| **Web服务层** | | | | |
| 前端 | `3000` | Frontend (React) | HTTP | Vite开发服务器 |
| 后端 | `3001` | Backend (Next.js) | HTTP | API服务器 |
| 监控面板 | `3002` | Grafana | HTTP | 监控界面 (可选) |
| **数据库层** | | | | |
| 主数据库 | `5432` | PostgreSQL | TCP | 关系型数据库 |
| 缓存 | `6379` | Redis | TCP | 内存缓存 |
| **消息队列层** | | | | |
| 消息队列 | `5672` | RabbitMQ | AMQP | 消息通信 |
| 管理界面 | `15672` | RabbitMQ UI | HTTP | 队列管理 |
| **AI服务层** | | | | |
| 文档处理 | `8001` | Document Processor | HTTP | Python微服务 |
| 向量化 | `8002` | Vector Service | HTTP | Python微服务 |
| RAG检索 | `8003` | RAG Service | HTTP | Python微服务 |
| **向量数据库层** | | | | |
| 向量数据库 | `19530` | Milvus | gRPC | 向量存储 |
| 管理界面 | `9091` | Milvus UI | HTTP | Attu管理界面 |
| **LLM服务层** | | | | |
| 语言模型 | `11434` | Ollama | HTTP | 本地LLM API |
| **监控服务层** | | | | |
| 监控收集 | `9092` | Prometheus | HTTP | 指标收集 |

## 🔧 开发环境配置

### 前端配置文件
```bash
# frontend/.env
VITE_API_BASE_URL=http://localhost:3001
VITE_APP_TITLE=RAG智能问答系统
VITE_APP_VERSION=1.0.0
VITE_ENABLE_DEVTOOLS=true
```

### 后端配置文件
```bash
# backend/.env
# 数据库连接
DATABASE_URL="postgresql://postgres:password@localhost:5432/rag_db"
REDIS_URL="redis://localhost:6379"
RABBITMQ_URL="amqp://guest:guest@localhost:5672"

# API服务
NEXTAUTH_URL="http://localhost:3001"
API_PORT="3001"

# Python微服务
DOCUMENT_PROCESSOR_URL="http://localhost:8001"
VECTOR_SERVICE_URL="http://localhost:8002" 
RAG_SERVICE_URL="http://localhost:8003"

# Milvus向量数据库
MILVUS_HOST="localhost"
MILVUS_PORT="19530"

# Ollama LLM
OLLAMA_BASE_URL="http://localhost:11434"
```

## 🐳 Docker Compose端口映射

```yaml
# 前端服务
frontend:
  ports:
    - "3000:3000"

# 后端服务
backend:
  ports:
    - "3001:3001"

# Python微服务
document-processor:
  ports:
    - "8001:8001"
vector-service:
  ports:
    - "8002:8002"
rag-service:
  ports:
    - "8003:8003"

# 数据库服务
postgres:
  ports:
    - "5432:5432"
redis:
  ports:
    - "6379:6379"
rabbitmq:
  ports:
    - "5672:5672"
    - "15672:15672"

# 向量数据库
milvus:
  ports:
    - "19530:19530"
    - "9091:9091"

# LLM服务
ollama:
  ports:
    - "11434:11434"
```

## 🔍 端口冲突检查

### 检查端口占用
```bash
# 检查关键端口是否被占用
ss -tlnp | grep -E ":(3000|3001|8001|8002|8003)"

# 或使用lsof
lsof -i :3000,3001,8001,8002,8003
```

### 清理端口占用
```bash
# 杀死指定端口进程
lsof -ti:3000,3001,3002,3003 | xargs kill -9

# 或单独杀死
kill -9 $(lsof -ti:3000)
```

## 🚀 启动顺序

### 开发环境启动
```bash
# 1. 启动后端服务 (3001端口)
cd backend && PORT=3001 npm run dev

# 2. 启动前端服务 (3000端口)
cd frontend && PORT=3000 npm run dev

# 3. 启动Python微服务 (可选)
cd python-services/rag-service && python main.py
```

### Docker环境启动
```bash
# 启动所有服务
docker-compose up -d

# 只启动核心服务 (前后端)
docker-compose up -d frontend backend postgres redis
```

## 📊 服务健康检查

### 检查服务状态
```bash
# 前端
curl http://localhost:3000

# 后端API
curl http://localhost:3001/api/health

# Python服务
curl http://localhost:8003/health

# 数据库连接
pg_isready -h localhost -p 5432 -U postgres

# Redis连接
redis-cli -h localhost -p 6379 ping
```

## 🔒 防火墙配置

### Ubuntu/Debian
```bash
# 开放必要端口
sudo ufw allow 3000  # 前端
sudo ufw allow 3001  # 后端
sudo ufw allow 5432  # PostgreSQL (仅本地)
sudo ufw allow 6379  # Redis (仅本地)
```

### 生产环境安全
- 仅对外开放 `3000` (前端) 和 `3001` (后端API)
- 数据库端口 `5432`, `6379` 仅内网访问
- Python微服务 `8001-8003` 仅集群内访问
- 管理界面 `15672`, `9091` 使用VPN或防火墙保护

---

**最后更新**: 2025-08-13  
**维护人员**: RAG系统开发团队  
**配置版本**: v1.0