from typing import Optional, Dict, Any, List
import ollama
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
import os
import logging
from config.config import get_env_value

# 配置日志
logger = logging.getLogger(__name__)

def get_env_int(key: str, default: str) -> int:
    """安全地获取并转换环境变量为整数"""
    value = get_env_value(key, default)
    try:
        return int(value)
    except (ValueError, AttributeError) as e:
        logger.error(f"Error parsing {key}={value}: {str(e)}")
        # 使用默认值
        return int(default)

def get_env_float(key: str, default: str) -> float:
    """安全地获取并转换环境变量为浮点数"""
    value = get_env_value(key, default)
    try:
        return float(value)
    except (ValueError, AttributeError) as e:
        logger.error(f"Error parsing {key}={value}: {str(e)}")
        # 使用默认值
        return float(default)

class OllamaClient(LLM):
    """Ollama LLM client that integrates with LangChain."""
    
    model_name: str = get_env_value("LLM_MODEL", "llama2")
    base_url: str = get_env_value("OLLAMA_BASE_URL", "http://localhost:11434")
    temperature: float = get_env_float("LLM_TEMPERATURE", "0.7")
    max_tokens: int = get_env_int("LLM_MAX_TOKENS", "2048")
    timeout: int = get_env_int("OLLAMA_TIMEOUT", "120")
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None
    ):
        super().__init__()
        if model_name:
            self.model_name = model_name
        if base_url:
            self.base_url = base_url
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if timeout is not None:
            self.timeout = timeout
            
        logger.info(f"Initialized OllamaClient with model: {self.model_name}")
    
    def _check_model_exists(self) -> bool:
        """检查模型是否已经在本地安装"""
        try:
            models = ollama.list()
            logger.debug(f"Available models: {models}")
            # 修复模型检查逻辑
            if 'models' in models and isinstance(models['models'], list):
                exists = any(model.get('model', '') == self.model_name for model in models['models'])
                if exists:
                    logger.info(f"Model {self.model_name} found locally")
                else:
                    logger.warning(f"Model {self.model_name} not found locally. Available models: {[model.get('model', '') for model in models['models']]}")
                return exists
            else:
                logger.error("Invalid response format from Ollama API")
                return False
        except Exception as e:
            logger.error(f"Error checking model existence: {str(e)}")
            return False
    
    @property
    def _llm_type(self) -> str:
        return "ollama"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """调用Ollama API生成回复"""
        try:
            # 确保temperature是浮点数
            if 'temperature' in kwargs:
                if not isinstance(kwargs['temperature'], (int, float)):
                    logger.warning(f"Invalid temperature value: {kwargs['temperature']}, using default: {self.temperature}")
                    kwargs['temperature'] = self.temperature
                kwargs['temperature'] = float(kwargs['temperature'])
            
            logger.debug(f"Calling Ollama API with params: {kwargs}")
            
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": kwargs.get('temperature', self.temperature),
                    "num_predict": kwargs.get('max_tokens', self.max_tokens),
                    "stop": stop if stop else []
                }
            )
            return response['response']
        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            raise
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """获取模型标识参数"""
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
    
    def load_model(self) -> None:
        """检查模型是否已安装，如果没有则提示用户安装"""
        logger.info(f"Checking if model {self.model_name} is installed...")
        if not self._check_model_exists():
            raise ValueError(
                f"Model {self.model_name} is not installed locally. "
                f"Please install it first using: ollama pull {self.model_name}"
            )
            
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        处理聊天消息列表并返回响应。
        
        Args:
            messages: 消息列表，每个消息应包含'role'和'content'字段
            
        Returns:
            str: 模型生成的回答
        """
        try:
            logger.debug(f"Calling Ollama chat API with {len(messages)} messages")
            
            # 将消息格式化为Ollama支持的格式
            formatted_messages = []
            for msg in messages:
                if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                    raise ValueError(f"Invalid message format: {msg}")
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # 调用Ollama chat API
            response = ollama.chat(
                model=self.model_name,
                messages=formatted_messages,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            )
            
            if isinstance(response, dict) and 'message' in response and 'content' in response['message']:
                return response['message']['content']
            else:
                logger.error(f"Unexpected response format from Ollama chat API: {response}")
                return str(response)
        except Exception as e:
            logger.error(f"Error calling Ollama chat API: {str(e)}")
            # 如果chat方法失败，尝试使用generate作为备选方案
            prompt = self._format_messages_as_prompt(messages)
            return self.generate(prompt)
            
    def _format_messages_as_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        将消息列表格式化为文本提示。
        
        Args:
            messages: 消息列表
            
        Returns:
            str: 格式化后的提示文本
        """
        prompt = ""
        for msg in messages:
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")
            prompt += f"{role}: {content}\n\n"
        
        # 添加Assistant前缀以生成响应
        prompt += "Assistant: "
        return prompt
        
    def generate(self, prompt: str) -> str:
        """
        生成文本响应的便捷方法。
        
        Args:
            prompt: 输入提示
            
        Returns:
            str: 生成的响应
        """
        return self._call(prompt) 