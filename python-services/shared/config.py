"""
配置管理模块
基于Pydantic的环境变量管理
"""

from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """
    应用配置设置
    从环境变量自动加载配置
    """
    
    # Milvus配置
    milvus_host: str = Field(default="localhost", description="Milvus服务器地址")
    milvus_port: int = Field(default=19530, description="Milvus端口")
    milvus_collection: str = Field(default="document_store", description="Milvus集合名称")
    milvus_timeout: int = Field(default=30, description="Milvus连接超时时间(秒)")
    
    # RabbitMQ配置
    rabbitmq_url: str = Field(
        default="amqp://guest:guest@localhost:5672", 
        description="RabbitMQ连接URL"
    )
    rabbitmq_exchange: str = Field(default="rag_exchange", description="RabbitMQ交换机名称")
    rabbitmq_document_queue: str = Field(
        default="document_processed", 
        description="文档处理队列"
    )
    rabbitmq_vector_queue: str = Field(
        default="vector_completed", 
        description="向量化完成队列"
    )
    
    # 向量化模型配置
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="嵌入模型名称"
    )
    embedding_dim: int = Field(default=2048, description="向量维度")
    embedding_batch_size: int = Field(default=32, description="批处理大小")
    max_seq_length: int = Field(default=256, description="最大序列长度")
    
    # Ollama配置
    ollama_base_url: str = Field(
        default="http://localhost:11434", 
        description="Ollama服务地址"
    )
    ollama_model: str = Field(default="llama2", description="Ollama模型名称")
    ollama_timeout: int = Field(default=30, description="Ollama请求超时时间")
    
    # 数据库配置 (用于日志记录等)
    database_url: Optional[str] = Field(
        default=None, 
        description="PostgreSQL数据库连接URL"
    )
    redis_url: Optional[str] = Field(
        default="redis://localhost:6379", 
        description="Redis连接URL"
    )
    
    # 服务配置
    document_processor_url: str = Field(
        default="http://localhost:8001",
        description="文档处理服务URL"
    )
    vector_service_url: str = Field(
        default="http://localhost:8002",
        description="向量化服务URL"
    )
    rag_service_url: str = Field(
        default="http://localhost:8003",
        description="RAG服务URL"
    )
    
    # 性能配置
    max_concurrent_requests: int = Field(default=10, description="最大并发请求数")
    request_timeout: int = Field(default=30, description="请求超时时间")
    max_document_size: int = Field(default=50 * 1024 * 1024, description="最大文档大小(字节)")
    
    # 日志配置
    log_level: str = Field(default="INFO", description="日志级别")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="日志格式"
    )
    enable_json_logs: bool = Field(default=True, description="启用JSON格式日志")
    
    # 缓存配置
    enable_cache: bool = Field(default=True, description="启用缓存")
    cache_ttl: int = Field(default=3600, description="缓存TTL(秒)")
    
    # 开发调试配置
    debug_mode: bool = Field(default=False, description="调试模式")
    enable_metrics: bool = Field(default=True, description="启用指标收集")
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = False
        
    def get_milvus_config(self) -> dict:
        """获取Milvus连接配置"""
        return {
            "host": self.milvus_host,
            "port": self.milvus_port,
            "timeout": self.milvus_timeout
        }
        
    def get_rabbitmq_config(self) -> dict:
        """获取RabbitMQ连接配置"""
        return {
            "url": self.rabbitmq_url,
            "exchange": self.rabbitmq_exchange,
            "queues": {
                "document_processed": self.rabbitmq_document_queue,
                "vector_completed": self.rabbitmq_vector_queue
            }
        }
        
    def get_embedding_config(self) -> dict:
        """获取嵌入模型配置"""
        return {
            "model_name": self.embedding_model,
            "dimension": self.embedding_dim,
            "batch_size": self.embedding_batch_size,
            "max_seq_length": self.max_seq_length
        }
        
    def get_ollama_config(self) -> dict:
        """获取Ollama配置"""
        return {
            "base_url": self.ollama_base_url,
            "model": self.ollama_model,
            "timeout": self.ollama_timeout
        }


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置实例"""
    return settings


def reload_settings():
    """重新加载配置"""
    global settings
    settings = Settings()
