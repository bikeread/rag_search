"""
向量存储工厂，用于创建不同类型的向量存储实例。
"""

import os
import logging
from typing import Optional, Dict, Any, Type

from langchain.embeddings.base import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from .base import VectorStoreBase, VectorStoreConfig, VectorStoreType
from .milvus_store import MilvusVectorStore

logger = logging.getLogger(__name__)

class VectorStoreFactory:
    """向量存储工厂类"""
    
    _stores = {
        "milvus": MilvusVectorStore,
        # 可以添加其他向量存储类型
    }
    
    @classmethod
    def create_store(cls, config: VectorStoreConfig) -> VectorStoreBase:
        """
        根据配置创建向量存储实例。

        Args:
            config (VectorStoreConfig): 向量存储配置

        Returns:
            VectorStoreBase: 向量存储实例
        """
        store_type = config.store_type.lower()
        store_class = cls._stores.get(store_type)
        
        if not store_class:
            raise ValueError(f"Unsupported vector store type: {store_type}")
            
        if store_type == "milvus":
            # 为 Milvus 存储创建特定的参数
            return store_class(
                collection_name=config.extra_args.get("collection_name", "default_collection"),
                embeddings=config.model_name,
                host=config.extra_args.get("host", "localhost"),
                port=config.extra_args.get("port", 19530)
            )
        else:
            # 对于其他存储类型，使用默认的配置参数
            return store_class(config)

    @staticmethod
    def create_store_with_embeddings(
        config: Optional[VectorStoreConfig] = None,
        embeddings: Optional[Embeddings] = None
    ) -> VectorStoreBase:
        """创建向量存储实例。
        
        Args:
            config: 向量存储配置
            embeddings: 嵌入模型实例
            
        Returns:
            向量存储实例
        """
        # 使用默认配置
        if config is None:
            config = VectorStoreConfig()
        
        # 创建嵌入模型
        if embeddings is None:
            logger.info(f"创建嵌入模型: {config.model_name}")
            embeddings = HuggingFaceEmbeddings(model_name=config.model_name)
        
        # 根据存储类型创建实例
        store_type = config.store_type.lower()
        
        if store_type == VectorStoreType.FAISS:
            logger.info("创建FAISS向量存储")
            from .faiss_store import FAISSVectorStore
            return FAISSVectorStore(config, embeddings)
            
        elif store_type == VectorStoreType.MILVUS:
            logger.info("创建Milvus向量存储")
            return MilvusVectorStore(config, embeddings)
            
        else:
            logger.warning(f"未知的向量存储类型: {store_type}，使用默认的FAISS")
            from .faiss_store import FAISSVectorStore
            return FAISSVectorStore(config, embeddings) 