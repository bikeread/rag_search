from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import os
import time
import asyncio
import aiohttp
import json
from datetime import datetime

class VectorServiceClient:
    """向量服务客户端"""
    
    def __init__(self, base_url: str = "http://vector-service:8002"):
        self.base_url = base_url
        
    async def search_vectors(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相似向量"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "query_text": query_text,
                    "top_k": top_k
                }
                
                async with session.post(
                    f"{self.base_url}/search",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("results", [])
                    else:
                        error_text = await response.text()
                        print(f"向量搜索失败: {response.status} - {error_text}")
                        return []
                        
        except Exception as e:
            print(f"调用向量服务失败: {str(e)}")
            return []

class OllamaClient:
    """Ollama LLM客户端"""
    
    def __init__(self, base_url: str = "http://ollama:11434"):
        self.base_url = base_url
        self.model = "llama3.2:latest"  # 使用轻量级模型
        
    async def generate_answer(self, context: str, query: str) -> str:
        """基于上下文生成回答"""
        try:
            prompt = f"""基于以下上下文信息，回答用户的问题。

上下文信息：
{context}

用户问题：{query}

请基于上述上下文信息简洁地回答问题。如果上下文中没有相关信息，请说明无法基于提供的信息回答。"""

            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # 低温度获得更一致的回答
                        "top_p": 0.9,
                        "max_tokens": 500
                    }
                }
                
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=60)  # 60秒超时
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "抱歉，无法生成回答。")
                    else:
                        error_text = await response.text()
                        print(f"LLM生成失败: {response.status} - {error_text}")
                        return "抱歉，LLM服务暂时不可用。"
                        
        except Exception as e:
            print(f"调用LLM服务失败: {str(e)}")
            return f"生成回答时出错：{str(e)}"

app = FastAPI(title="RAG Service", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局客户端实例
vector_client = VectorServiceClient()
llm_client = OllamaClient()

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
    query_time: float
    status: str
    metadata: Dict[str, Any] = {}

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
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

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """执行RAG查询"""
    start_time = time.time()
    
    try:
        query = request.query
        top_k = request.top_k
        
        print(f"🔍 开始RAG查询: '{query}' (top_k={top_k})")
        
        # 1. 向量检索阶段
        print("📋 第1阶段：向量检索")
        search_results = await vector_client.search_vectors(query, top_k)
        
        if not search_results:
            print("⚠️  未找到相关文档")
            return QueryResponse(
                answer="抱歉，我无法在文档库中找到与您问题相关的信息。",
                sources=[],
                query_time=time.time() - start_time,
                status="no_results",
                metadata={"search_results_count": 0}
            )
        
        print(f"✅ 找到 {len(search_results)} 个相关文档片段")
        
        # 2. 构建上下文
        print("📝 第2阶段：构建上下文")
        contexts = []
        sources = []
        
        for i, result in enumerate(search_results):
            text = result.get("text", "").strip()
            if text:  # 只处理非空文本
                contexts.append(f"文档片段{i+1}：{text}")
                
                sources.append(SourceDocument(
                    id=result.get("id", f"doc_{i}"),
                    score=result.get("score", 0.0),
                    text=text,
                    metadata=result.get("metadata", {})
                ))
        
        if not contexts:
            print("⚠️  检索到的文档片段为空")
            return QueryResponse(
                answer="检索到的文档片段内容为空，无法生成回答。",
                sources=sources,
                query_time=time.time() - start_time,
                status="empty_context",
                metadata={"search_results_count": len(search_results)}
            )
        
        context = "\n\n".join(contexts)
        print(f"📄 上下文构建完成，总长度: {len(context)} 字符")
        
        # 3. LLM生成阶段
        print("🤖 第3阶段：LLM答案生成")
        answer = await llm_client.generate_answer(context, query)
        
        query_time = time.time() - start_time
        print(f"✅ RAG查询完成，耗时: {query_time:.2f}秒")
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            query_time=query_time,
            status="completed",
            metadata={
                "search_results_count": len(search_results),
                "context_length": len(context),
                "sources_count": len(sources)
            }
        )
        
    except Exception as e:
        query_time = time.time() - start_time
        print(f"❌ RAG查询失败: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"RAG查询失败: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)