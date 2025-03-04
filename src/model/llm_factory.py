from typing import Optional
from langchain.llms.base import LLM
import os
import logging
from src.model.ollama_client import OllamaClient
from config.config import get_env_value

# 配置日志
logger = logging.getLogger(__name__)

def get_env_value(key: str, default: str) -> str:
    """安全地获取环境变量值，移除注释和空白"""
    value = os.getenv(key, default)
    if value:
        # 移除注释和空白
        value = value.split('#')[0].strip()
    return value or default

class LLMFactory:
    """工厂类，用于创建LLM实例"""
    
    @staticmethod
    def create_llm() -> LLM:
        """
        根据环境配置创建合适的LLM实例
        """
        provider = get_env_value("LLM_PROVIDER", "huggingface").lower()
        logger.debug(f"Creating LLM with provider: {provider}")
        
        if provider == "ollama":
            return OllamaClient()
        elif provider == "huggingface":
            # TODO: 实现HuggingFace模型支持
            raise NotImplementedError("HuggingFace support not implemented yet")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
            
    @staticmethod
    def get_llm_instance(
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> LLM:
        """
        获取LLM实例，支持自定义配置
        
        Args:
            provider: LLM提供者，如果不指定则使用环境变量配置
            model_name: 模型名称，如果不指定则使用环境变量配置
            **kwargs: 其他配置参数
        """
        if provider is None:
            provider = get_env_value("LLM_PROVIDER", "huggingface").lower()
            
        if provider == "ollama":
            return OllamaClient(model_name=model_name, **kwargs)
        elif provider == "huggingface":
            # TODO: 实现HuggingFace模型支持
            raise NotImplementedError("HuggingFace support not implemented yet")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}") 