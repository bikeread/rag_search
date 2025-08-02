from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

app = FastAPI(title="Document Processor Service", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "document-processor"}

@app.post("/process-document")
async def process_document(file: UploadFile = File(...)):
    """处理上传的文档"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # 读取文件内容
        content = await file.read()
        
        # TODO: 集成现有的文档处理逻辑
        # 这里暂时返回模拟数据
        
        return {
            "status": "processing",
            "filename": file.filename,
            "size": len(content),
            "mime_type": file.content_type,
            "chunks_count": 5  # 模拟分块数量
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)