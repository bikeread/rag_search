"""
Milvus客户端
统一的Milvus向量数据库客户端，供所有服务使用
"""

import asyncio
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pymilvus import Collection, connections, utility, FieldSchema, CollectionSchema, DataType
import numpy as np
import time

from .config import settings
from .utils.exceptions import MilvusConnectionError, VectorizationError
from .logging_config import get_logger

logger = get_logger(__name__)


class MilvusClient:
    """
    Milvus客户端
    提供向量存储、检索、管理功能
    """
    
    def __init__(self, collection_name: str = None):
        """
        初始化Milvus客户端
        
        Args:
            collection_name: 集合名称，默认从配置获取
        """
        self.collection_name = collection_name or settings.milvus_collection
        self.collection: Optional[Collection] = None
        self.is_connected = False
        self.connection_alias = "default"
        
        # 从配置获取连接参数
        self.host = settings.milvus_host
        self.port = settings.milvus_port
        self.timeout = settings.milvus_timeout
        
        logger.info(f"Initializing Milvus client for collection: {self.collection_name}")
    
    async def connect(self) -> bool:
        """
        建立Milvus连接
        
        Returns:
            bool: 连接是否成功
            
        Raises:
            MilvusConnectionError: 连接失败时抛出
        """
        try:
            logger.info(f"Connecting to Milvus at {self.host}:{self.port}")
            
            # 建立连接
            connections.connect(
                alias=self.connection_alias,
                host=self.host,
                port=self.port,
                timeout=self.timeout
            )
            
            # 验证连接
            if not connections.has_connection(self.connection_alias):
                raise MilvusConnectionError("Failed to establish connection")
            
            # 初始化集合
            await self._initialize_collection()
            
            self.is_connected = True
            logger.info("Successfully connected to Milvus")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise MilvusConnectionError(f"Connection failed: {str(e)}")
    
    async def _initialize_collection(self):
        """初始化集合"""
        try:
            # 检查集合是否存在
            if utility.has_collection(self.collection_name):
                logger.info(f"Collection {self.collection_name} already exists")
                self.collection = Collection(self.collection_name)
            else:
                logger.info(f"Creating collection {self.collection_name}")
                self.collection = await self._create_collection()
            
            # 确保集合已加载
            if not self.collection.has_index():
                await self._create_index()
            
            # 加载集合到内存
            self.collection.load()
            logger.info(f"Collection {self.collection_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize collection: {str(e)}")
            raise MilvusConnectionError(f"Collection initialization failed: {str(e)}")
    
    async def _create_collection(self) -> Collection:
        """
        创建集合
        
        Returns:
            Collection: 创建的集合对象
        """
        # 定义字段Schema
        fields = [
            FieldSchema(
                name="id", 
                dtype=DataType.VARCHAR,
                max_length=100,
                is_primary=True,
                description="主键ID"
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=65535,
                description="原始文本内容"
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=settings.embedding_dim,
                description="文本向量"
            ),
            FieldSchema(
                name="document_id",
                dtype=DataType.VARCHAR,
                max_length=100,
                description="文档ID"
            ),
            FieldSchema(
                name="chunk_index",
                dtype=DataType.INT64,
                description="文档块索引"
            ),
            FieldSchema(
                name="created_at",
                dtype=DataType.INT64,
                description="创建时间戳"
            )
        ]
        
        # 创建Schema
        schema = CollectionSchema(
            fields=fields,
            description=f"RAG系统文档向量存储集合"
        )
        
        # 创建集合
        collection = Collection(
            name=self.collection_name,
            schema=schema,
            using=self.connection_alias
        )
        
        logger.info(f"Created collection {self.collection_name} with {len(fields)} fields")
        return collection
    
    async def _create_index(self):
        """创建向量索引"""
        try:
            # 向量索引参数
            index_params = {
                "metric_type": "COSINE",  # 余弦相似度
                "index_type": "IVF_FLAT",  # IVF_FLAT索引
                "params": {"nlist": 128}  # 聚类中心数量
            }
            
            logger.info("Creating vector index...")
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            # 等待索引创建完成
            while True:
                indexing_progress = utility.index_building_progress(self.collection_name)
                if indexing_progress["pending_index_rows"] == 0:
                    break
                await asyncio.sleep(1)
            
            logger.info("Vector index created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            raise MilvusConnectionError(f"Index creation failed: {str(e)}")
    
    async def upsert_vectors(self, 
                           vectors: List[List[float]], 
                           texts: List[str], 
                           document_id: str,
                           metadata: List[Dict] = None) -> List[str]:
        """
        批量插入或更新向量
        
        Args:
            vectors: 向量列表
            texts: 对应的文本列表
            document_id: 文档ID
            metadata: 元数据列表
            
        Returns:
            List[str]: 插入的向量ID列表
            
        Raises:
            VectorizationError: 插入失败时抛出
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            # 生成唯一ID
            vector_ids = [str(uuid.uuid4()) for _ in texts]
            current_timestamp = int(time.time())
            
            # 准备插入数据
            entities = [
                vector_ids,  # id字段
                texts,       # text字段
                vectors,     # embedding字段
                [document_id] * len(texts),  # document_id字段
                list(range(len(texts))),     # chunk_index字段
                [current_timestamp] * len(texts)  # created_at字段
            ]
            
            # 调试信息
            logger.info(f"Inserting {len(vectors)} vectors for document {document_id}")
            logger.info(f"Sample text data: {texts[:2] if texts else 'None'}")
            logger.info(f"Entities structure: [ids({len(vector_ids)}), texts({len(texts)}), vectors({len(vectors)}), doc_ids({len([document_id] * len(texts))}), indexes({len(list(range(len(texts))))}), timestamps({len([current_timestamp] * len(texts))})]")
            
            # 执行插入
            insert_result = self.collection.insert(entities)
            
            # 刷新确保数据持久化
            self.collection.flush()
            
            logger.info(f"Successfully inserted {len(vector_ids)} vectors")
            return vector_ids
            
        except Exception as e:
            logger.error(f"Failed to insert vectors: {str(e)}")
            raise VectorizationError(f"Vector insertion failed: {str(e)}")
    
    async def search_vectors(self, 
                           query_vector: List[float], 
                           top_k: int = 5,
                           filters: Dict = None) -> List[Dict]:
        """
        向量相似性搜索
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            filters: 搜索过滤条件
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            # 搜索参数
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}  # 搜索的聚类中心数量
            }
            
            # 构建搜索表达式
            expr = None
            if filters:
                conditions = []
                if "document_id" in filters:
                    conditions.append(f'document_id == "{filters["document_id"]}"')
                if conditions:
                    expr = " and ".join(conditions)
            
            logger.debug(f"Searching for {top_k} similar vectors")
            
            # 执行搜索
            search_results = self.collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=["id", "text", "document_id", "chunk_index", "created_at"]
            )
            
            # 格式化结果
            formatted_results = []
            for hits in search_results:
                for hit in hits:
                    # 调试信息：查看返回的实体字段
                    logger.info(f"Debug: hit.entity fields: {list(hit.entity.keys()) if hasattr(hit.entity, 'keys') else 'No keys'}")
                    logger.info(f"Debug: hit.entity.text = '{hit.entity.get('text')}'")
                    
                    result = {
                        "id": hit.id,
                        "score": float(hit.score),
                        "text": hit.entity.get("text"),
                        "document_id": hit.entity.get("document_id"),
                        "chunk_index": hit.entity.get("chunk_index"),
                        "created_at": hit.entity.get("created_at"),
                        "metadata": {}
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} similar vectors")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise VectorizationError(f"Search failed: {str(e)}")
    
    async def delete_vectors(self, vector_ids: List[str]) -> bool:
        """
        删除指定向量
        
        Args:
            vector_ids: 要删除的向量ID列表
            
        Returns:
            bool: 删除是否成功
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            # 构建删除表达式
            ids_str = ", ".join([f'"{vid}"' for vid in vector_ids])
            expr = f"id in [{ids_str}]"
            
            logger.info(f"Deleting {len(vector_ids)} vectors")
            
            # 执行删除
            self.collection.delete(expr)
            self.collection.flush()
            
            logger.info("Vectors deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {str(e)}")
            return False
    
    async def delete_document_vectors(self, document_id: str) -> bool:
        """
        删除文档的所有向量
        
        Args:
            document_id: 文档ID
            
        Returns:
            bool: 删除是否成功
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            expr = f'document_id == "{document_id}"'
            
            logger.info(f"Deleting all vectors for document {document_id}")
            
            # 执行删除
            self.collection.delete(expr)
            self.collection.flush()
            
            logger.info(f"All vectors for document {document_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document vectors: {str(e)}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息
        
        Returns:
            Dict: 统计信息
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            stats = {
                "name": self.collection_name,
                "num_entities": self.collection.num_entities,
                "is_loaded": self.collection.is_loaded,
                "has_index": self.collection.has_index()
            }
            
            # 获取索引信息
            if self.collection.has_index():
                index_info = self.collection.index()
                stats["index_info"] = {
                    "metric_type": index_info.params.get("metric_type"),
                    "index_type": index_info.params.get("index_type")
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {"error": str(e)}
    
    async def close(self):
        """关闭连接"""
        try:
            if self.collection:
                self.collection.release()
            
            if connections.has_connection(self.connection_alias):
                connections.disconnect(self.connection_alias)
            
            self.is_connected = False
            logger.info("Milvus connection closed")
            
        except Exception as e:
            logger.error(f"Error closing Milvus connection: {str(e)}")
    
    def __del__(self):
        """析构函数"""
        if self.is_connected:
            try:
                # 注意：这里不能使用async，所以只是标记状态
                self.is_connected = False
            except:
                pass


# 全局客户端实例（可选）
_global_client: Optional[MilvusClient] = None


async def get_milvus_client(collection_name: str = None) -> MilvusClient:
    """
    获取Milvus客户端实例
    
    Args:
        collection_name: 集合名称
        
    Returns:
        MilvusClient: 客户端实例
    """
    global _global_client
    
    if _global_client is None or not _global_client.is_connected:
        _global_client = MilvusClient(collection_name)
        await _global_client.connect()
    
    return _global_client
