"""
TF-IDF向量化器实现。
"""

import os
import pickle
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer

from .base import BaseVectorizer

logger = logging.getLogger(__name__)

class TFIDFVectorizer(BaseVectorizer):
    """基于TF-IDF的向量化器实现。"""
    
    def __init__(self, max_features: int = 100):
        """初始化TF-IDF向量化器。
        
        Args:
            max_features: 最大特征数量，即向量维度
        """
        self.max_features = max_features
        self.vectorizer = SklearnTfidfVectorizer(
            max_features=self.max_features,
            lowercase=True,
            analyzer='word',
            stop_words='english'  # 仅针对英文文本
        )
        self.is_fitted = False
        logger.info(f"初始化TF-IDF向量化器，最大特征数：{max_features}")
    
    def fit(self, texts: List[str]) -> "TFIDFVectorizer":
        """使用文本列表训练向量化器。
        
        Args:
            texts: 文本列表
            
        Returns:
            返回自身实例，方便链式调用
        """
        logger.info(f"训练TF-IDF向量化器，使用{len(texts)}个文本")
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self
    
    def transform(self, texts: List[str]) -> List[List[float]]:
        """将文本列表转换为向量列表。
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表
        """
        if not self.is_fitted:
            logger.warning("TF-IDF向量化器尚未训练，先执行训练")
            return self.fit_transform(texts)
        
        logger.debug(f"转换{len(texts)}个文本为TF-IDF向量")
        sparse_vectors = self.vectorizer.transform(texts)
        return sparse_vectors.toarray().tolist()
    
    def save(self, path: str) -> bool:
        """保存向量化器到指定路径。
        
        Args:
            path: 保存路径
            
        Returns:
            是否保存成功
        """
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            logger.info(f"TF-IDF向量化器已保存到{path}")
            return True
        except Exception as e:
            logger.error(f"保存TF-IDF向量化器失败: {str(e)}")
            return False
    
    @classmethod
    def load(cls, path: str) -> "TFIDFVectorizer":
        """从指定路径加载向量化器。
        
        Args:
            path: 加载路径
            
        Returns:
            加载的向量化器实例
        """
        try:
            with open(path, 'rb') as f:
                instance = cls()
                instance.vectorizer = pickle.load(f)
                instance.is_fitted = True
                logger.info(f"从{path}加载TF-IDF向量化器")
                return instance
        except Exception as e:
            logger.error(f"加载TF-IDF向量化器失败: {str(e)}")
            return cls()  # 返回新实例 