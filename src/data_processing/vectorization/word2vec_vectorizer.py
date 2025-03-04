"""
Word2Vec向量化器模块，使用Word2Vec算法将文本转换为向量。
"""

import os
import logging
import numpy as np
from typing import List, Optional, Dict, Any
import gensim
from gensim.models import Word2Vec, KeyedVectors
import re
import jieba

from .base import BaseVectorizer

logger = logging.getLogger(__name__)

class Word2VecVectorizer(BaseVectorizer):
    """Word2Vec向量化器，使用Word2Vec算法将文本转换为向量。"""
    
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1, 
                 workers: int = 4, pretrained_path: str = '', cache_dir: str = './cache/vectorization'):
        """初始化Word2Vec向量化器。
        
        Args:
            vector_size: 向量维度
            window: 窗口大小
            min_count: 最小词频
            workers: 工作线程数
            pretrained_path: 预训练模型路径，如果提供则加载预训练模型
            cache_dir: 缓存目录
        """
        super().__init__(cache_dir)
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.pretrained_path = pretrained_path
        
        # 初始化模型
        self.model = None
        self.is_fitted = False
        
        # 加载预训练模型或创建新模型
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained_model(pretrained_path)
        else:
            self._load_or_create_model()
    
    def _load_pretrained_model(self, path: str):
        """加载预训练模型。
        
        Args:
            path: 预训练模型路径
        """
        try:
            # 尝试加载不同格式的预训练模型
            if path.endswith('.bin'):
                self.model = KeyedVectors.load_word2vec_format(path, binary=True)
            elif path.endswith('.txt') or path.endswith('.vec'):
                self.model = KeyedVectors.load_word2vec_format(path, binary=False)
            else:
                self.model = Word2Vec.load(path)
            
            if isinstance(self.model, Word2Vec):
                self.model = self.model.wv
                
            self.vector_size = self.model.vector_size
            self.is_fitted = True
            logger.info(f"已加载预训练Word2Vec模型: {path}")
        except Exception as e:
            logger.error(f"加载预训练Word2Vec模型失败: {str(e)}")
            self._create_new_model()
    
    def _load_or_create_model(self):
        """加载或创建模型。"""
        model_path = os.path.join(self.cache_dir, "word2vec_model.pkl")
        if os.path.exists(model_path):
            try:
                loaded_vectorizer = self.load(model_path)
                self.model = loaded_vectorizer.model
                self.is_fitted = loaded_vectorizer.is_fitted
                self.vector_size = loaded_vectorizer.vector_size
                logger.info("已加载现有Word2Vec模型")
            except Exception as e:
                logger.warning(f"加载Word2Vec模型失败，将创建新模型: {str(e)}")
                self._create_new_model()
        else:
            self._create_new_model()
    
    def _create_new_model(self):
        """创建新模型。"""
        self.model = None
        self.is_fitted = False
    
    def _preprocess_text(self, text: str) -> List[str]:
        """预处理文本。
        
        Args:
            text: 要处理的文本
            
        Returns:
            分词后的词列表
        """
        # 清理文本
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # 检测是否包含中文
        if re.search(r'[\u4e00-\u9fff]', text):
            # 中文分词
            return list(jieba.cut(text))
        else:
            # 英文分词
            return text.split()
    
    def _ensure_fitted(self, texts: List[str]):
        """确保模型已经训练。
        
        Args:
            texts: 用于训练的文本列表
        """
        if not self.is_fitted:
            logger.info("Word2Vec模型未训练，开始训练...")
            
            # 预处理文本
            tokenized_texts = [self._preprocess_text(text) for text in texts]
            
            # 训练模型
            self.model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                workers=self.workers
            ).wv
            
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
            
            # 分词
            words = self._preprocess_text(text)
            
            # 计算词向量平均值
            word_vectors = []
            for word in words:
                try:
                    if word in self.model:
                        word_vectors.append(self.model[word])
                except:
                    continue
            
            if word_vectors:
                return np.mean(word_vectors, axis=0)
            else:
                return np.zeros(self.vector_size)
            
        except Exception as e:
            logger.error(f"Word2Vec向量化失败: {str(e)}")
            return np.zeros(self.vector_size)
    
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
            
            # 处理每个文本
            vectors = []
            for text in texts:
                vectors.append(self.vectorize(text))
            
            return vectors
            
        except Exception as e:
            logger.error(f"Word2Vec批量向量化失败: {str(e)}")
            return [np.zeros(self.vector_size) for _ in texts]
    
    def get_most_similar(self, word: str, topn: int = 10) -> List[Dict[str, Any]]:
        """获取与给定词最相似的词。
        
        Args:
            word: 查询词
            topn: 返回结果数量
            
        Returns:
            相似词列表，每个元素包含词和相似度
        """
        if not self.is_fitted:
            raise ValueError("Word2Vec模型尚未训练")
        
        try:
            similar_words = self.model.most_similar(word, topn=topn)
            return [{"word": word, "similarity": similarity} for word, similarity in similar_words]
        except KeyError:
            logger.warning(f"词 '{word}' 不在词汇表中")
            return []
            
    def get_embedding_dim(self) -> int:
        """获取嵌入向量的维度。
        
        Returns:
            向量维度
        """
        return self.vector_size 