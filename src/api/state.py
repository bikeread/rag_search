"""
Application state management.
"""
from typing import Optional
from src.rag.retriever import RAGRetriever
from src.model.llm_factory import LLMFactory
from langchain.llms.base import LLM

class AppState:
    """Application state singleton."""
    _instance = None
    
    def __init__(self):
        self.llm: Optional[LLM] = None
        self.retriever: Optional[RAGRetriever] = None
    
    @classmethod
    def get_instance(cls) -> 'AppState':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = AppState()
        return cls._instance
    
    def initialize(self, llm: Optional[LLM] = None):
        """Initialize application components."""
        # 设置LLM
        self.llm = llm or LLMFactory.create_llm()
        
        # 初始化基础检索器和RAG检索器
        from src.data_processing.storage.factory import VectorStoreFactory
        from src.data_processing.storage.base import VectorStoreConfig, VectorStoreType
        from src.rag.retriever import VectorStoreRetriever, RAGRetriever
        
        # 创建向量存储配置
        from config.config import (
            VECTOR_DB_TYPE, 
            VECTOR_DB_PATH, 
            EMBEDDING_MODEL,
            MILVUS_HOST,
            MILVUS_PORT,
            MILVUS_COLLECTION
        )
        
        # 根据配置选择存储类型
        store_type = "milvus" if VECTOR_DB_TYPE.lower() == 'milvus' else "faiss"
        
        # 创建向量存储配置
        store_config = VectorStoreConfig(
            store_type=store_type,
            model_name=EMBEDDING_MODEL,
            extra_args={
                "host": MILVUS_HOST,
                "port": MILVUS_PORT,
                "collection_name": MILVUS_COLLECTION
            } if store_type == "milvus" else {"vector_db_path": VECTOR_DB_PATH}
        )
        
        # 创建向量存储
        vector_store = VectorStoreFactory.create_store(store_config)
        
        # 创建RAG检索器，传入必要的参数
        self.retriever = RAGRetriever(
            vector_store=vector_store,
            llm=self.llm,
            query_enhancement_enabled=True,
            chain_type="qa"
        )
        
        return True

# 全局状态实例
app_state = AppState.get_instance() 