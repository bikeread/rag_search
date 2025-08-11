# RAG系统 API 文档

## 认证接口

### POST /api/auth/signin
用户登录认证接口。

**请求参数**:
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**响应格式**:
```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "user": {
    "id": "user-123",
    "email": "user@example.com",
    "name": "张三"
  }
}
```

**错误码**:
- `401`: 认证失败，用户名或密码错误
- `400`: 请求参数格式错误
- `429`: 请求过于频繁，请稍后重试

## 文档管理接口

### POST /api/documents/upload
上传文档到系统进行处理。

**请求格式**: multipart/form-data

**参数说明**:
- `file`: 文档文件 (必需)
- `title`: 文档标题 (可选)
- `tags`: 标签列表，逗号分隔 (可选)
- `description`: 文档描述 (可选)

**支持格式**:
- PDF (.pdf) - 最大50MB
- Word (.docx, .doc) - 最大30MB
- Excel (.xlsx, .xls) - 最大20MB
- Text (.txt, .md) - 最大10MB

**响应示例**:
```json
{
  "document_id": "doc-456",
  "status": "processing",
  "estimated_time": 30,
  "chunks_count": 0
}
```

### GET /api/documents/list
获取用户的文档列表。

**查询参数**:
- `page`: 页码，默认1
- `limit`: 每页数量，默认20，最大100
- `status`: 筛选状态 (processing/completed/failed)
- `search`: 搜索关键词

**响应格式**:
```json
{
  "documents": [
    {
      "id": "doc-123",
      "title": "技术规范文档",
      "upload_time": "2024-08-11T10:30:00Z",
      "status": "completed",
      "chunks_count": 45,
      "file_size": 2048576,
      "tags": ["技术", "规范"]
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 156,
    "pages": 8
  }
}
```

### DELETE /api/documents/{id}
删除指定文档及其所有相关数据。

**路径参数**:
- `id`: 文档ID

**响应**:
```json
{
  "success": true,
  "message": "文档已成功删除",
  "deleted_chunks": 45
}
```

## RAG查询接口

### POST /api/query
执行RAG智能查询。

**请求参数**:
```json
{
  "query": "如何配置系统认证？",
  "top_k": 5,
  "document_ids": ["doc-123", "doc-456"],
  "temperature": 0.7
}
```

**参数说明**:
- `query`: 查询问题 (必需)
- `top_k`: 返回相关片段数量，默认3
- `document_ids`: 限定查询的文档ID列表 (可选)
- `temperature`: 生成温度，0-1之间，默认0.7

**响应格式**:
```json
{
  "answer": "根据文档，系统认证配置步骤如下...",
  "sources": [
    {
      "document_id": "doc-123",
      "document_title": "系统配置手册",
      "chunk_text": "认证配置章节内容...",
      "relevance_score": 0.92,
      "page": 15
    }
  ],
  "query_time": 2.3,
  "tokens_used": 450
}
```

### GET /api/query/history
获取查询历史记录。

**查询参数**:
- `page`: 页码
- `limit`: 每页数量
- `start_date`: 开始日期
- `end_date`: 结束日期

## 错误处理

所有API在发生错误时返回统一格式：

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "错误描述信息",
    "details": {
      "field": "具体错误字段",
      "reason": "详细原因"
    }
  }
}
```

## 速率限制

- 认证接口: 5次/分钟
- 上传接口: 10次/小时
- 查询接口: 60次/分钟
- 其他接口: 100次/分钟

超出限制时返回429状态码，响应头包含：
- `X-RateLimit-Limit`: 限制数量
- `X-RateLimit-Remaining`: 剩余次数
- `X-RateLimit-Reset`: 重置时间戳