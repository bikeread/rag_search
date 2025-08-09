"""
文档处理服务
FastAPI服务入口，处理文档上传和处理请求
"""

import asyncio
import uuid
import os
import sys
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import aiohttp
import json

# 添加共享组件路径
# sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))  # 临时注释，文件已复制到本地

from config import settings
from logging_config import setup_logging
from rabbit_client import RabbitMQClient, ProcessingStatus
from utils.exceptions import DocumentProcessingError
from utils.validators import validate_document

# 导入文档处理器
from src.document_processor import ModernDocumentProcessor

# 设置日志
logger = setup_logging("document-processor")

# 创建FastAPI应用
app = FastAPI(
    title="Document Processor Service",
    description="文档处理微服务 - 处理各种格式文档的上传、解析和分块",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VectorServiceClient:
    """向量化服务客户端"""
    
    def __init__(self, base_url: str = "http://vector-service:8002"):
        self.base_url = base_url
        
    async def vectorize_and_store(self, texts: List[str], document_id: str) -> Dict[str, Any]:
        """向量化文本并存储到Milvus"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "texts": texts,
                    "store_vectors": True,
                    "document_ids": [document_id] * len(texts)
                }
                
                async with session.post(
                    f"{self.base_url}/vectorize",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"成功向量化 {len(texts)} 个文本块，存储了 {result.get('stored_count', 0)} 个向量")
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"向量化服务调用失败: {response.status} - {error_text}")
                        return {"status": "failed", "error": error_text}
                        
        except Exception as e:
            logger.error(f"调用向量化服务失败: {str(e)}")
            return {"status": "failed", "error": str(e)}

# 全局变量
rabbit_client: Optional[RabbitMQClient] = None
document_processor: Optional[ModernDocumentProcessor] = None
vector_client = VectorServiceClient()


# Pydantic模型定义
class DocumentProcessRequest(BaseModel):
    """文档处理请求模型"""
    document_id: Optional[str] = Field(None, description="文档ID，如果不提供将自动生成")
    filename: str = Field(..., description="文件名")
    enable_chunking: bool = Field(True, description="是否启用文本分块")
    chunk_size: int = Field(1000, description="分块大小")
    chunk_overlap: int = Field(200, description="分块重叠")
    metadata: Optional[Dict[str, Any]] = Field(None, description="额外元数据")


class DocumentChunk(BaseModel):
    """文档分块模型"""
    chunk_id: str = Field(..., description="分块ID")
    content: str = Field(..., description="分块内容")
    chunk_index: int = Field(..., description="分块索引")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="分块元数据")


class ProcessingResult(BaseModel):
    """处理结果模型"""
    document_id: str = Field(..., description="文档ID")
    filename: str = Field(..., description="文件名")
    status: str = Field(..., description="处理状态")
    chunks: List[DocumentChunk] = Field(default_factory=list, description="文档分块列表")
    chunk_count: int = Field(0, description="分块数量")
    processing_time: float = Field(0.0, description="处理耗时（秒）")
    error_message: Optional[str] = Field(None, description="错误信息")


class ProcessingStatusResponse(BaseModel):
    """处理状态响应模型"""
    document_id: str = Field(..., description="文档ID")
    status: str = Field(..., description="当前状态")
    progress: float = Field(0.0, description="处理进度 (0-100)")
    message: str = Field("", description="状态描述")
    created_at: str = Field(..., description="创建时间")
    updated_at: str = Field(..., description="更新时间")


# 依赖注入
async def get_rabbit_client() -> RabbitMQClient:
    """获取RabbitMQ客户端"""
    global rabbit_client
    if rabbit_client is None or not rabbit_client.is_connected:
        rabbit_client = RabbitMQClient()
        await rabbit_client.connect()
    return rabbit_client


# 生命周期事件
@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    logger.info("🚀 文档处理服务启动中...")
    
    try:
        # 初始化RabbitMQ连接
        global rabbit_client, document_processor
        rabbit_client = RabbitMQClient()
        await rabbit_client.connect()
        logger.info("✅ RabbitMQ连接成功")
        
        # 初始化文档处理器
        document_processor = ModernDocumentProcessor()
        logger.info("✅ 文档处理器初始化成功")
        
        # 初始化其他组件
        logger.info("✅ 文档处理服务启动完成")
        
    except Exception as e:
        logger.error(f"❌ 服务启动失败: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的清理"""
    logger.info("🔄 文档处理服务关闭中...")
    
    try:
        # 清理RabbitMQ连接
        if rabbit_client:
            await rabbit_client.close()
        
        logger.info("✅ 文档处理服务关闭完成")
        
    except Exception as e:
        logger.error(f"❌ 服务关闭时出错: {str(e)}")


# API路由定义
@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        # 检查RabbitMQ连接
        rabbit_status = "healthy" if rabbit_client and rabbit_client.is_connected else "unhealthy"
        
        return {
            "status": "healthy",
            "service": "document-processor",
            "version": "1.0.0",
            "components": {
                "rabbitmq": rabbit_status
            },
            "capabilities": [
                "文档上传处理",
                "多格式文档解析",
                "文本分块",
                "异步处理通知"
            ]
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/process-document", response_model=ProcessingResult)
async def process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_id: Optional[str] = None,
    enable_chunking: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    rabbit_client: RabbitMQClient = Depends(get_rabbit_client)
):
    """
    处理上传的文档
    
    Args:
        background_tasks: FastAPI后台任务
        file: 上传的文件
        document_id: 可选的文档ID
        enable_chunking: 是否启用分块
        chunk_size: 分块大小
        chunk_overlap: 分块重叠
        rabbit_client: RabbitMQ客户端
        
    Returns:
        ProcessingResult: 处理结果
    """
    start_time = asyncio.get_event_loop().time()
    
    # 生成文档ID
    if not document_id:
        document_id = str(uuid.uuid4())
    
    logger.info(f"开始处理文档: {file.filename}, 文档ID: {document_id}")
    
    try:
        # 读取文件内容
        file_content = await file.read()
        
        # 验证文档
        validate_document(file.filename, file_content, file.content_type)
        
        # 发布处理开始消息
        await rabbit_client.publish_processing_status(
            document_id=document_id,
            status=ProcessingStatus.PROCESSING,
            details={
                "filename": file.filename,
                "file_size": len(file_content),
                "content_type": file.content_type,
                "enable_chunking": enable_chunking
            }
        )
        
        # 添加后台处理任务
        background_tasks.add_task(
            process_document_async,
            document_id=document_id,
            filename=file.filename,
            file_content=file_content,
            content_type=file.content_type,
            enable_chunking=enable_chunking,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            rabbit_client=rabbit_client
        )
        
        processing_time = asyncio.get_event_loop().time() - start_time
        
        return ProcessingResult(
            document_id=document_id,
            filename=file.filename,
            status=ProcessingStatus.PROCESSING,
            chunks=[],
            chunk_count=0,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"文档处理失败: {str(e)}")
        
        # 发布失败消息
        await rabbit_client.publish_processing_status(
            document_id=document_id,
            status=ProcessingStatus.FAILED,
            details={"error": str(e)}
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"文档处理失败: {str(e)}"
        )


@app.get("/processing-status/{document_id}", response_model=ProcessingStatusResponse)
async def get_processing_status(document_id: str):
    """
    获取文档处理状态
    
    Args:
        document_id: 文档ID
        
    Returns:
        ProcessingStatusResponse: 处理状态
    """
    # TODO: 实现状态查询逻辑，可以从Redis或数据库获取
    # 暂时返回基本状态
    return ProcessingStatusResponse(
        document_id=document_id,
        status="processing",
        progress=50.0,
        message="处理中...",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z"
    )


@app.post("/reprocess-document/{document_id}")
async def reprocess_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    rabbit_client: RabbitMQClient = Depends(get_rabbit_client)
):
    """
    重新处理文档
    
    Args:
        document_id: 文档ID
        background_tasks: 后台任务
        rabbit_client: RabbitMQ客户端
        
    Returns:
        dict: 操作结果
    """
    logger.info(f"重新处理文档: {document_id}")
    
    try:
        # TODO: 实现重新处理逻辑
        # 1. 从存储中获取原始文档
        # 2. 重新执行处理流程
        
        await rabbit_client.publish_processing_status(
            document_id=document_id,
            status=ProcessingStatus.PROCESSING,
            details={"action": "reprocess"}
        )
        
        return {
            "status": "success",
            "message": f"文档 {document_id} 已加入重新处理队列",
            "document_id": document_id
        }
        
    except Exception as e:
        logger.error(f"重新处理文档失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"重新处理失败: {str(e)}"
        )


# 后台处理函数
async def process_document_async(
    document_id: str,
    filename: str,
    file_content: bytes,
    content_type: str,
    enable_chunking: bool,
    chunk_size: int,
    chunk_overlap: int,
    rabbit_client: RabbitMQClient
):
    """
    异步处理文档的后台任务
    
    Args:
        document_id: 文档ID
        filename: 文件名
        file_content: 文件内容
        content_type: 内容类型
        enable_chunking: 是否启用分块
        chunk_size: 分块大小
        chunk_overlap: 分块重叠
        rabbit_client: RabbitMQ客户端
    """
    try:
        logger.info(f"开始异步处理文档: {filename}")
        
        # 使用文档处理器处理文件
        result = await document_processor.process_file(
            file_content=file_content,
            filename=filename,
            mime_type=content_type,
            document_id=document_id,
            enable_chunking=enable_chunking,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        if result.status == "failed":
            raise Exception(result.error_message)
        
        chunks = result.chunks
        
        # 提取文本进行向量化
        texts = [chunk.content for chunk in chunks if chunk.content.strip()]
        vector_result = {"status": "skipped", "stored_count": 0}
        
        if texts:
            logger.info(f"开始向量化 {len(texts)} 个文本块...")
            vector_result = await vector_client.vectorize_and_store(texts, document_id)
        
        # 发布处理完成消息
        await rabbit_client.publish_document_processed({
            "document_id": document_id,
            "filename": filename,
            "status": ProcessingStatus.COMPLETED,
            "chunks": [chunk.dict() for chunk in chunks],
            "chunk_count": len(chunks),
            "vectorization": {
                "status": vector_result.get("status", "failed"),
                "vector_count": vector_result.get("stored_count", 0),
                "vector_ids": vector_result.get("vector_ids", [])
            },
            "metadata": {
                "content_type": content_type,
                "file_size": len(file_content),
                "enable_chunking": enable_chunking,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
        })
        
        logger.info(f"文档处理完成: {filename}, 生成 {len(chunks)} 个分块")
        
    except Exception as e:
        logger.error(f"异步处理文档失败: {str(e)}")
        
        # 发布失败消息
        await rabbit_client.publish_document_processed({
            "document_id": document_id,
            "filename": filename,
            "status": ProcessingStatus.FAILED,
            "error": str(e),
            "chunks": [],
            "chunk_count": 0
        })


# 主函数
if __name__ == "__main__":
    # 从环境变量获取配置
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    logger.info(f"启动文档处理服务: {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )