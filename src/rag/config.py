"""
RAG系统配置模块。
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class RAGConfig:
    """RAG配置类，用于管理检索增强生成相关配置。"""
    
    # 向量存储配置
    vector_store_type: str = field(default_factory=lambda: os.getenv("VECTOR_STORE_TYPE", "faiss"))
    connection_args: Dict[str, Any] = field(default_factory=dict)
    
    # 文档处理配置
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")))
    
    # 检索配置
    top_k_results: int = field(default_factory=lambda: int(os.getenv("TOP_K_RESULTS", "4")))
    similarity_threshold: float = field(default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.7")))
    
    # 语言模型配置
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-3.5-turbo"))
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "2000")))
    
    # 嵌入模型配置
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    
    # 查询优化配置
    query_enhancement_enabled: bool = field(
        default_factory=lambda: os.getenv("QUERY_ENHANCEMENT_ENABLED", "true").lower() == "true"
    )
    query_decomposition_enabled: bool = field(
        default_factory=lambda: os.getenv("QUERY_DECOMPOSITION_ENABLED", "true").lower() == "true"
    )
    query_expansion_enabled: bool = field(
        default_factory=lambda: os.getenv("QUERY_EXPANSION_ENABLED", "true").lower() == "true"
    )
    keyword_extraction_enabled: bool = field(
        default_factory=lambda: os.getenv("KEYWORD_EXTRACTION_ENABLED", "true").lower() == "true"
    )
    keyword_search_enabled: bool = field(
        default_factory=lambda: os.getenv("KEYWORD_SEARCH_ENABLED", "true").lower() == "true"
    )
    max_enhanced_docs: int = field(
        default_factory=lambda: int(os.getenv("MAX_ENHANCED_DOCS", "10"))
    )
    
    # 中文处理配置
    enable_chinese_processing: bool = field(
        default_factory=lambda: os.getenv("ENABLE_CHINESE_PROCESSING", "true").lower() == "true"
    )
    
    def __post_init__(self):
        """初始化后的处理，设置默认的连接参数等。"""
        # 根据向量存储类型设置默认连接参数
        if not self.connection_args:
            if self.vector_store_type.lower() == "faiss":
                self.connection_args = {"vector_db_path": os.getenv("VECTOR_DB_PATH", "data/vector_store.faiss")}
            elif self.vector_store_type.lower() == "milvus":
                self.connection_args = {
                    "host": os.getenv("MILVUS_HOST", "localhost"),
                    "port": os.getenv("MILVUS_PORT", "19530"),
                    "collection_name": os.getenv("MILVUS_COLLECTION", "documents")
                }
                
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典格式。"""
        return {
            "vector_store_type": self.vector_store_type,
            "connection_args": self.connection_args,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k_results": self.top_k_results,
            "similarity_threshold": self.similarity_threshold,
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "embedding_model": self.embedding_model,
            "enable_chinese_processing": self.enable_chinese_processing,
            "query_enhancement_enabled": self.query_enhancement_enabled,
            "query_decomposition_enabled": self.query_decomposition_enabled,
            "query_expansion_enabled": self.query_expansion_enabled,
            "keyword_extraction_enabled": self.keyword_extraction_enabled,
            "keyword_search_enabled": self.keyword_search_enabled,
            "max_enhanced_docs": self.max_enhanced_docs
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        """从字典创建配置对象。"""
        return cls(**config_dict) 