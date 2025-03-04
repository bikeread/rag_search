"""
向量化基类模块，定义向量化器的接口。
"""

import os
import logging
import pickle
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Any
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseVectorizer(ABC):
    """向量化器基类，定义向量化器的接口。"""
    
    def __init__(self, cache_dir: str = './cache/vectorization'):
        """初始化向量化器。
        
        Args:
            cache_dir: 缓存目录，用于保存模型
        """
        self.cache_dir = cache_dir
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """确保缓存目录存在。"""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    @abstractmethod
    def vectorize(self, text: str) -> np.ndarray:
        """将文本转换为向量。
        
        Args:
            text: 要向量化的文本
            
        Returns:
            文本的向量表示
        """
        pass
    
    @abstractmethod
    def batch_vectorize(self, texts: List[str]) -> List[np.ndarray]:
        """批量将文本转换为向量。
        
        Args:
            texts: 要向量化的文本列表
            
        Returns:
            文本的向量表示列表
        """
        pass
    
    def save(self, path: Optional[str] = None) -> str:
        """保存向量化器模型。
        
        Args:
            path: 保存路径，如果为None则使用默认路径
            
        Returns:
            实际保存的路径
        """
        if path is None:
            path = os.path.join(self.cache_dir, f"{self.__class__.__name__}.pkl")
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"向量化器已保存到: {path}")
            return path
        except Exception as e:
            logger.error(f"保存向量化器失败: {str(e)}")
            raise
    
    @classmethod
    def load(cls, path: str) -> 'BaseVectorizer':
        """加载向量化器模型。
        
        Args:
            path: 模型路径
            
        Returns:
            加载的向量化器实例
            
        Raises:
            FileNotFoundError: 如果模型文件不存在
            ValueError: 如果加载的模型类型不匹配
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")
        
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            
            if not isinstance(model, cls):
                raise ValueError(f"加载的模型类型不匹配，期望 {cls.__name__}，实际 {model.__class__.__name__}")
            
            logger.info(f"向量化器已从 {path} 加载")
            return model
        except Exception as e:
            logger.error(f"加载向量化器失败: {str(e)}")
            raise
    
    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量之间的余弦相似度。
        
        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            
        Returns:
            余弦相似度，范围为[-1, 1]
        """
        # 确保向量是一维的
        if len(vec1.shape) > 1:
            vec1 = vec1.flatten()
        if len(vec2.shape) > 1:
            vec2 = vec2.flatten()
        
        # 计算余弦相似度
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2) 