"""
向量存储基类及通用配置。
"""

import os
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from langchain.schema import Document
from langchain.embeddings.base import Embeddings
from .config import VectorStoreConfig

logger = logging.getLogger(__name__)

class VectorStoreType(str, Enum):
    """向量存储类型枚举。"""
    
    FAISS = "faiss"
    MILVUS = "milvus"

@dataclass
class VectorStoreConfig:
    """向量存储配置类。"""
    
    # 存储类型
    store_type: str = field(default_factory=lambda: os.getenv('VECTOR_DB_TYPE', 'faiss'))
    
    # 存储路径
    store_path: str = field(default_factory=lambda: os.getenv('VECTOR_DB_PATH', 'data/vector_store'))
    
    # 嵌入模型名称（默认使用中文优化模型）
    model_name: str = field(default_factory=lambda: os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3'))
    
    # 相似度阈值
    min_similarity_score: float = field(default_factory=lambda: float(os.getenv('MIN_SIMILARITY_SCORE', '0.05')))
    
    # 最大返回结果数
    top_k: int = field(default_factory=lambda: int(os.getenv('TOP_K_RESULTS', '3')))
    
    # 索引参数
    index_params: Dict[str, Any] = field(default_factory=dict)
    
    # 连接参数（如Milvus连接配置）
    connection_args: Dict[str, Any] = field(default_factory=dict)
    
    # 额外参数（存储各种配置选项）
    extra_args: Dict[str, Any] = field(default_factory=dict)
    
    # 文档类型，方便预设参数
    doc_type: Optional[str] = None

class VectorStoreBase(ABC):
    """
    向量存储基类。
    """
    
    def __init__(self, config: VectorStoreConfig):
        """
        初始化向量存储。

        Args:
            config (VectorStoreConfig): 向量存储配置
        """
        self.config = config
        
    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        搜索相似文档。

        Args:
            query (str): 查询文本
            k (int, optional): 返回结果数量. 默认为 5.

        Returns:
            List[Dict[str, Any]]: 相似文档列表
        """
        pass
        
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        添加文档。

        Args:
            documents (List[Dict[str, Any]]): 文档列表

        Returns:
            bool: 是否添加成功
        """
        pass
        
    @abstractmethod
    def delete_documents(self, ids: List[str]) -> bool:
        """
        删除文档。

        Args:
            ids (List[str]): 文档ID列表

        Returns:
            bool: 是否删除成功
        """
        pass

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> bool:
        """添加文本到向量存储。
        
        Args:
            texts: 要添加的文本列表
            metadatas: 可选的元数据列表
            
        Returns:
            是否成功添加
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def clear(self) -> bool:
        """清空向量存储。
        
        Returns:
            是否成功清空
        """
        raise NotImplementedError("子类必须实现此方法")
        
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息
        
        Returns:
            包含统计信息的字典，如文档数量、向量维度等
        """
        return {"error": "该向量存储不支持统计信息查询"} 