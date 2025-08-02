from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import os

app = FastAPI(title="RAG Service", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class SourceDocument(BaseModel):
    id: str
    score: float
    text: str
    metadata: Dict[str, Any] = {}

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    query_time: int
    status: str

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "rag-service"}

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """执行RAG查询"""
    try:
        query = request.query
        top_k = request.top_k
        
        # TODO: 集成现有的RAG检索逻辑
        # 这里暂时返回模拟数据
        
        mock_sources = [
            SourceDocument(
                id=f"doc_{i}",
                score=0.9 - i * 0.1,
                text=f"这是文档{i}的相关内容，与查询'{query}'高度相关。",
                metadata={"filename": f"document_{i}.pdf", "page": i+1}
            )
            for i in range(top_k)
        ]
        
        mock_answer = f"基于检索到的文档内容，针对您的问题'{query}'，我的回答是：这是一个示例回答，需要集成真实的LLM服务。"
        
        return QueryResponse(
            answer=mock_answer,
            sources=mock_sources,
            query_time=1500,  # 模拟1.5秒响应时间
            status="completed"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)