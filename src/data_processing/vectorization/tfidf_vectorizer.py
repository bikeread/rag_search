"""
TF-IDF向量化器模块，使用TF-IDF算法将文本转换为向量。
"""

import os
import logging
import numpy as np
from typing import List, Optional, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidfVectorizer
from sklearn.exceptions import NotFittedError

from .base import BaseVectorizer

logger = logging.getLogger(__name__)

class TfidfVectorizer(BaseVectorizer):
    """TF-IDF向量化器，使用TF-IDF算法将文本转换为向量。"""
    
    def __init__(self, max_features: int = 5000, use_idf: bool = True, cache_dir: str = './cache/vectorization'):
        """初始化TF-IDF向量化器。
        
        Args:
            max_features: 最大特征数量
            use_idf: 是否使用IDF权重
            cache_dir: 缓存目录
        """
        super().__init__(cache_dir)
        self.max_features = max_features
        self.use_idf = use_idf
        
        # 初始化sklearn的TfidfVectorizer
        self.vectorizer = SklearnTfidfVectorizer(
            max_features=max_features,
            use_idf=use_idf,
            analyzer='word',
            token_pattern=r'\w{1,}',
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.is_fitted = False
        self._load_or_create_model()
    
    def _load_or_create_model(self):
        """加载或创建模型。"""
        model_path = os.path.join(self.cache_dir, "tfidf_model.pkl")
        if os.path.exists(model_path):
            try:
                loaded_vectorizer = self.load(model_path)
                self.vectorizer = loaded_vectorizer.vectorizer
                self.is_fitted = loaded_vectorizer.is_fitted
                logger.info("已加载现有TF-IDF模型")
            except Exception as e:
                logger.warning(f"加载TF-IDF模型失败，将创建新模型: {str(e)}")
                self.is_fitted = False
    
    def _ensure_fitted(self, texts: List[str]):
        """确保模型已经训练。
        
        Args:
            texts: 用于训练的文本列表
        """
        if not self.is_fitted:
            logger.info("TF-IDF模型未训练，开始训练...")
            self.vectorizer.fit(texts)
            self.is_fitted = True
            self.save()
    
    def vectorize(self, text: str) -> np.ndarray:
        """将文本转换为向量。
        
        Args:
            text: 要向量化的文本
            
        Returns:
            文本的向量表示
        """
        try:
            # 确保模型已训练
            self._ensure_fitted([text])
            
            # 转换文本
            vector = self.vectorizer.transform([text]).toarray()[0]
            return vector
            
        except Exception as e:
            logger.error(f"TF-IDF向量化失败: {str(e)}")
            # 返回零向量
            return np.zeros(self.max_features)
    
    def batch_vectorize(self, texts: List[str]) -> List[np.ndarray]:
        """批量将文本转换为向量。
        
        Args:
            texts: 要向量化的文本列表
            
        Returns:
            文本的向量表示列表
        """
        try:
            # 确保模型已训练
            self._ensure_fitted(texts)
            
            # 转换文本
            vectors = self.vectorizer.transform(texts).toarray()
            return [vector for vector in vectors]
            
        except Exception as e:
            logger.error(f"TF-IDF批量向量化失败: {str(e)}")
            # 返回零向量列表
            return [np.zeros(self.max_features) for _ in texts]
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称（词汇表）。
        
        Returns:
            特征名称列表
        """
        if not self.is_fitted:
            raise NotFittedError("TF-IDF模型尚未训练，无法获取特征名称")
        
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_embedding_dim(self) -> int:
        """获取嵌入向量的维度。
        
        Returns:
            向量维度
        """
        return self.max_features 