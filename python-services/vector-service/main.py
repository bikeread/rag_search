from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import os
import uuid
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import json
import asyncio
from datetime import datetime

# 导入shared模块
from config import settings
from logging_config import setup_logging, get_logger
from milvus_client import MilvusClient
from utils.exceptions import VectorServiceError

# 设置日志
setup_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Vector Service", 
    version="1.0.0",
    description="向量化服务 - 负责文本向量化和向量存储"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CPUVectorizer:
    """CPU友好的轻量级向量化器，适用于AMD处理器"""
    
    def __init__(self):
        # 使用TF-IDF作为基础向量化方法，适配小数据集优化
        self.tfidf = TfidfVectorizer(
            max_features=384,  # 固定维度，便于存储
            stop_words=None,  # 保留所有词，包括中文
            ngram_range=(1, 1),  # 只使用单词，降低复杂性
            min_df=1,         # 最少出现1次
            max_df=1.0,       # 允许所有词，避免小数据集max_df<min_df问题
            token_pattern=r'[^\s]+',  # 支持中文分词
            lowercase=True,   # 统一转为小写提高匹配率
            sublinear_tf=True,  # 使用对数TF，减少高频词影响
            smooth_idf=True,   # 平滑IDF，避免零除错误
            norm='l2'         # L2归一化
        )
        self.is_fitted = False
        self.target_dim = 384  # 目标向量维度
        self.vocabulary_base = []  # 基础词汇表
        
    def vectorize_texts(self, texts: List[str]) -> np.ndarray:
        """将文本转换为向量"""
        if not texts:
            return np.zeros((0, self.target_dim))
            
        # 预处理文本，确保不为空
        processed_texts = []
        for text in texts:
            if not text or not text.strip():
                processed_texts.append("empty_text_placeholder")
            else:
                processed_texts.append(text.strip())
        
        if not self.is_fitted:
            # 首次使用时拟合模型
            vectors = self.tfidf.fit_transform(processed_texts)
            self.is_fitted = True
            logger.info(f"TF-IDF向量化器已拟合，词汇表大小: {len(self.tfidf.vocabulary_)}")
        else:
            vectors = self.tfidf.transform(processed_texts)
        
        # 转换为密集数组
        dense_vectors = vectors.toarray()
        logger.info(f"原始向量维度: {dense_vectors.shape}, 非零元素: {np.count_nonzero(dense_vectors)}")
        
        # 确保向量维度为384
        current_dim = dense_vectors.shape[1]
        if current_dim < self.target_dim:
            # 如果维度不足，用小的随机数填充而不是零
            padding = np.random.normal(0, 0.01, (dense_vectors.shape[0], self.target_dim - current_dim))
            dense_vectors = np.hstack([dense_vectors, padding])
        elif current_dim > self.target_dim:
            # 如果维度过多，截取前384维
            dense_vectors = dense_vectors[:, :self.target_dim]
        
        # 防止全零向量
        for i, vec in enumerate(dense_vectors):
            if np.allclose(vec, 0):
                # 为全零向量添加小的随机扰动
                dense_vectors[i] = np.random.normal(0, 0.01, self.target_dim)
                logger.warning(f"向量 {i} 为全零，已添加随机扰动")
        
        # 归一化
        normalized_vectors = normalize(dense_vectors, norm='l2')
        
        logger.info(f"最终向量维度: {normalized_vectors.shape}, 向量范数: {[f'{np.linalg.norm(v):.3f}' for v in normalized_vectors[:3]]}")
        
        return normalized_vectors

# 全局实例
vectorizer = CPUVectorizer()
milvus_client: Optional[MilvusClient] = None

# 数据模型
class VectorizeRequest(BaseModel):
    texts: List[str] = Field(..., description="待向量化的文本列表")
    document_ids: Optional[List[str]] = Field(None, description="文档ID列表")
    metadata: Optional[List[Dict[str, Any]]] = Field(None, description="元数据列表")
    collection_name: Optional[str] = Field("rag_documents", description="Milvus集合名称")
    store_vectors: bool = Field(True, description="是否存储向量到Milvus")

class VectorizeResponse(BaseModel):
    vector_ids: List[str] = Field(..., description="向量ID列表")
    vectors: List[List[float]] = Field(..., description="向量列表")
    status: str = Field(..., description="处理状态")
    stored_count: int = Field(0, description="存储到Milvus的向量数量")

class SearchRequest(BaseModel):
    query_text: Optional[str] = Field(None, description="查询文本")
    query_vector: Optional[List[float]] = Field(None, description="查询向量")
    collection_name: str = Field("rag_documents", description="搜索的集合名称") 
    top_k: int = Field(5, description="返回最相似的向量数量")
    filter_expr: Optional[str] = Field(None, description="过滤表达式")

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="搜索结果")
    query_vector: Optional[List[float]] = Field(None, description="查询向量")
    status: str = Field(..., description="搜索状态")
    search_time: float = Field(..., description="搜索耗时")

class StoreVectorRequest(BaseModel):
    vectors: List[List[float]] = Field(..., description="向量列表")
    vector_ids: List[str] = Field(..., description="向量ID列表")
    metadata: Optional[List[Dict[str, Any]]] = Field(None, description="元数据列表")
    collection_name: str = Field("rag_documents", description="集合名称")

# 启动和关闭事件
@app.on_event("startup")
async def startup_event():
    """服务启动事件"""
    global milvus_client
    
    logger.info("🚀 向量化服务启动中...")
    
    try:
        # 初始化Milvus客户端
        milvus_client = MilvusClient()
        await milvus_client.connect()
        logger.info("✅ Milvus连接成功")
        
        # 集合在连接时已自动初始化
        logger.info("✅ 默认集合初始化完成")
        
    except Exception as e:
        logger.error(f"❌ Milvus连接失败: {str(e)}")
        # 不抛出异常，允许服务在没有Milvus的情况下运行
    
    logger.info("✅ 向量化服务启动完成")

@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭事件"""
    global milvus_client
    
    logger.info("🔄 向量化服务关闭中...")
    
    if milvus_client:
        try:
            await milvus_client.close()
            logger.info("✅ Milvus连接已关闭")
        except Exception as e:
            logger.error(f"❌ 关闭Milvus连接失败: {str(e)}")
    
    logger.info("✅ 向量化服务已关闭")

@app.get("/health")
async def health_check():
    """健康检查"""
    milvus_status = "disconnected"
    if milvus_client and milvus_client.is_connected:
        milvus_status = "connected"
    
    return {
        "status": "healthy",
        "service": "vector-service",
        "version": "1.0.0",
        "vectorizer": "CPU-TF-IDF",
        "components": {
            "milvus": milvus_status
        },
        "capabilities": [
            "文本向量化",
            "向量存储",
            "向量搜索",
            "批量处理"
        ]
    }

@app.post("/vectorize", response_model=VectorizeResponse)
async def vectorize_texts(request: VectorizeRequest):
    """向量化文本并可选择存储到Milvus"""
    try:
        texts = request.texts
        if not texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        logger.info(f"开始向量化 {len(texts)} 个文本")
        
        # 生成向量
        vectors = vectorizer.vectorize_texts(texts)
        
        # 生成向量ID（如果未提供）
        if request.document_ids:
            if len(request.document_ids) != len(texts):
                raise HTTPException(status_code=400, detail="document_ids length must match texts length")
            vector_ids = request.document_ids
        else:
            vector_ids = [str(uuid.uuid4()) for _ in texts]
        
        # 转换为列表格式
        vector_list = [vec.tolist() for vec in vectors]
        
        stored_count = 0
        
        # 存储到Milvus（如果启用）
        if request.store_vectors and milvus_client and milvus_client.is_connected:
            try:
                # 准备元数据
                metadata_list = request.metadata or [{} for _ in texts]
                if len(metadata_list) != len(texts):
                    metadata_list = metadata_list[:len(texts)] + [{} for _ in range(len(texts) - len(metadata_list))]
                
                # 为每个向量添加文本内容到元数据
                for i, meta in enumerate(metadata_list):
                    meta.update({
                        "text": texts[i],
                        "vector_id": vector_ids[i],
                        "created_at": datetime.now().isoformat()
                    })
                
                # 存储向量 - 使用第一个文档ID作为batch标识
                document_id = request.document_ids[0] if request.document_ids else "batch_" + str(uuid.uuid4())[:8]
                
                await milvus_client.upsert_vectors(
                    vectors=vector_list,
                    texts=texts,
                    document_id=document_id,
                    metadata=metadata_list
                )
                stored_count = len(vector_list)
                logger.info(f"成功存储 {stored_count} 个向量到Milvus")
                
            except Exception as e:
                logger.error(f"存储向量到Milvus失败: {str(e)}")
                # 不抛出异常，允许返回向量而不存储
        
        return VectorizeResponse(
            vector_ids=vector_ids,
            vectors=vector_list,
            status="completed",
            stored_count=stored_count
        )
        
    except Exception as e:
        logger.error(f"向量化失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Vectorization failed: {str(e)}")

@app.post("/query-vectorize")
async def query_vectorize(request: dict):
    """为查询文本生成向量"""
    try:
        query_text = request.get("query", "")
        if not query_text:
            raise HTTPException(status_code=400, detail="No query text provided")
        
        logger.info(f"为查询文本生成向量: {query_text[:100]}...")
        
        # 为单个查询生成向量
        vectors = vectorizer.vectorize_texts([query_text])
        query_vector = vectors[0].tolist()
        
        return {
            "query_vector": query_vector,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"查询向量化失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query vectorization failed: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_vectors(request: SearchRequest):
    """向量搜索"""
    import time
    start_time = time.time()
    
    try:
        # 获取查询向量
        if request.query_text:
            # 从文本生成向量
            logger.info(f"为查询文本生成向量: {request.query_text[:100]}...")
            vectors = vectorizer.vectorize_texts([request.query_text])
            query_vector = vectors[0].tolist()
        elif request.query_vector:
            # 使用提供的向量
            query_vector = request.query_vector
        else:
            raise HTTPException(status_code=400, detail="Either query_text or query_vector must be provided")
        
        results = []
        
        # 执行Milvus搜索
        if milvus_client and milvus_client.is_connected:
            try:
                logger.info(f"在集合中搜索相似向量...")
                search_results = await milvus_client.search_vectors(
                    query_vector=query_vector,
                    top_k=request.top_k,
                    filters=None  # TODO: 实现过滤器转换
                )
                
                # 格式化搜索结果
                for result in search_results:
                    results.append({
                        "id": result.get("id"),
                        "score": result.get("score", 0.0),
                        "text": result.get("text", ""),
                        "document_id": result.get("document_id", ""),
                        "metadata": result.get("metadata", {})
                    })
                
                logger.info(f"搜索完成，找到 {len(results)} 个结果")
                
            except Exception as e:
                logger.error(f"Milvus搜索失败: {str(e)}")
                # 继续执行，返回空结果
        else:
            logger.warning("Milvus未连接，无法执行搜索")
        
        search_time = time.time() - start_time
        
        return SearchResponse(
            results=results,
            query_vector=query_vector,
            status="completed",
            search_time=search_time
        )
        
    except Exception as e:
        logger.error(f"向量搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")

@app.post("/store-vectors")
async def store_vectors(request: StoreVectorRequest):
    """直接存储向量到Milvus"""
    try:
        if not request.vectors or not request.vector_ids:
            raise HTTPException(status_code=400, detail="Vectors and vector_ids are required")
        
        if len(request.vectors) != len(request.vector_ids):
            raise HTTPException(status_code=400, detail="Vectors and vector_ids length must match")
        
        if not milvus_client or not milvus_client.is_connected:
            raise HTTPException(status_code=503, detail="Milvus is not connected")
        
        logger.info(f"存储 {len(request.vectors)} 个向量到Milvus")
        
        # 准备元数据
        metadata_list = request.metadata or [{} for _ in request.vectors]
        if len(metadata_list) != len(request.vectors):
            metadata_list = metadata_list[:len(request.vectors)] + [{} for _ in range(len(request.vectors) - len(metadata_list))]
        
        # 为每个向量添加基本信息到元数据
        for i, meta in enumerate(metadata_list):
            meta.update({
                "vector_id": request.vector_ids[i],
                "created_at": datetime.now().isoformat()
            })
        
        # 存储向量 - 需要提供texts参数
        texts = [meta.get('text', f'Vector {i}') for i, meta in enumerate(metadata_list)]
        document_id = f"manual_store_{str(uuid.uuid4())[:8]}"
        
        await milvus_client.upsert_vectors(
            vectors=request.vectors,
            texts=texts,
            document_id=document_id,
            metadata=metadata_list
        )
        
        logger.info(f"成功存储 {len(request.vectors)} 个向量")
        
        return {
            "status": "completed",
            "stored_count": len(request.vectors),
            "collection_name": request.collection_name
        }
        
    except Exception as e:
        logger.error(f"存储向量失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Store vectors failed: {str(e)}")

@app.delete("/document/{document_id}")
async def delete_document_vectors(document_id: str):
    """删除指定文档的所有向量"""
    try:
        if not document_id:
            raise HTTPException(status_code=400, detail="Document ID is required")
        
        if not milvus_client or not milvus_client.is_connected:
            logger.warning("Milvus未连接，跳过向量删除")
            return {
                "status": "skipped",
                "message": "Milvus not connected, vector deletion skipped",
                "document_id": document_id
            }
        
        logger.info(f"删除文档 {document_id} 的所有向量")
        
        # 执行删除
        success = await milvus_client.delete_document_vectors(document_id)
        
        if success:
            logger.info(f"成功删除文档 {document_id} 的向量")
            return {
                "status": "completed",
                "message": "Document vectors deleted successfully",
                "document_id": document_id
            }
        else:
            logger.error(f"删除文档 {document_id} 的向量失败")
            raise HTTPException(status_code=500, detail="Failed to delete document vectors")
        
    except Exception as e:
        logger.error(f"删除文档向量失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Delete document vectors failed: {str(e)}")

if __name__ == "__main__":
    logger.info("启动向量化服务...")
    uvicorn.run(app, host="0.0.0.0", port=8002)