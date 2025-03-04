"""
Word2Vec向量化器实现。
"""

import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from gensim.models import Word2Vec
import jieba

from .base import BaseVectorizer

logger = logging.getLogger(__name__)

class Word2VecVectorizer(BaseVectorizer):
    """基于Word2Vec的向量化器实现。"""
    
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1):
        """初始化Word2Vec向量化器。
        
        Args:
            vector_size: 向量维度
            window: 上下文窗口大小
            min_count: 词语最小出现次数
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
        self.is_fitted = False
        self.is_chinese = os.getenv('DEFAULT_LANGUAGE', 'zh') == 'zh'
        logger.info(f"初始化Word2Vec向量化器，向量维度：{vector_size}，窗口大小：{window}")
    
    def _tokenize(self, text: str) -> List[str]:
        """将文本分词。
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果
        """
        if self.is_chinese:
            return list(jieba.cut(text))
        else:
            return text.lower().split()
    
    def fit(self, texts: List[str]) -> "Word2VecVectorizer":
        """使用文本列表训练向量化器。
        
        Args:
            texts: 文本列表
            
        Returns:
            返回自身实例，方便链式调用
        """
        logger.info(f"训练Word2Vec向量化器，使用{len(texts)}个文本")
        # 对每个文本进行分词
        tokenized_texts = [self._tokenize(text) for text in texts]
        
        # 训练Word2Vec模型
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4
        )
        
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
            logger.warning("Word2Vec向量化器尚未训练，先执行训练")
            return self.fit_transform(texts)
        
        logger.debug(f"转换{len(texts)}个文本为Word2Vec向量")
        
        result = []
        for text in texts:
            tokens = self._tokenize(text)
            vectors = []
            
            # 获取每个词的向量
            for token in tokens:
                if token in self.model.wv:
                    vectors.append(self.model.wv[token])
            
            if vectors:
                # 计算平均向量
                avg_vector = np.mean(vectors, axis=0).tolist()
                result.append(avg_vector)
            else:
                # 如果没有词向量，返回零向量
                result.append([0.0] * self.vector_size)
        
        return result
    
    def save(self, path: str) -> bool:
        """保存向量化器到指定路径。
        
        Args:
            path: 保存路径
            
        Returns:
            是否保存成功
        """
        try:
            self.model.save(path)
            logger.info(f"Word2Vec向量化器已保存到{path}")
            return True
        except Exception as e:
            logger.error(f"保存Word2Vec向量化器失败: {str(e)}")
            return False
    
    @classmethod
    def load(cls, path: str) -> "Word2VecVectorizer":
        """从指定路径加载向量化器。
        
        Args:
            path: 加载路径
            
        Returns:
            加载的向量化器实例
        """
        try:
            instance = cls()
            instance.model = Word2Vec.load(path)
            instance.vector_size = instance.model.vector_size
            instance.is_fitted = True
            logger.info(f"从{path}加载Word2Vec向量化器")
            return instance
        except Exception as e:
            logger.error(f"加载Word2Vec向量化器失败: {str(e)}")
            return cls()  # 返回新实例 