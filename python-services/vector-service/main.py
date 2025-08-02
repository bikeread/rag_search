from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import os

app = FastAPI(title="Vector Service", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VectorizeRequest(BaseModel):
    texts: List[str]

class VectorizeResponse(BaseModel):
    vector_ids: List[str]
    status: str

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "vector-service"}

@app.post("/vectorize", response_model=VectorizeResponse)
async def vectorize_texts(request: VectorizeRequest):
    """向量化文本并存储到Milvus"""
    try:
        texts = request.texts
        
        # TODO: 集成现有的向量化逻辑和Milvus存储
        # 这里暂时返回模拟数据
        
        mock_vector_ids = [f"vec_{i}" for i in range(len(texts))]
        
        return VectorizeResponse(
            vector_ids=mock_vector_ids,
            status="completed"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)