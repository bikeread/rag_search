"""
BERT向量化器实现。
"""

import os
import logging
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

from .base import BaseVectorizer

logger = logging.getLogger(__name__)

class BERTVectorizer(BaseVectorizer):
    """基于BERT的向量化器实现。"""
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512):
        """初始化BERT向量化器。
        
        Args:
            model_name: BERT模型名称
            max_length: 最大文本长度
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            logger.info(f"加载BERT模型: {model_name}, 设备: {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.is_fitted = True
        except Exception as e:
            logger.error(f"加载BERT模型失败: {str(e)}")
            self.tokenizer = None
            self.model = None
            self.is_fitted = False
    
    def fit(self, texts: List[str]) -> "BERTVectorizer":
        """BERT模型不需要额外训练，此方法仅作兼容。
        
        Args:
            texts: 文本列表
            
        Returns:
            返回自身实例，方便链式调用
        """
        logger.info("BERT向量化器无需训练")
        return self
    
    def transform(self, texts: List[str]) -> List[List[float]]:
        """将文本列表转换为向量列表。
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表
        """
        if not self.is_fitted:
            logger.error("BERT模型未正确加载，无法进行向量化")
            return [[0.0] * 768 for _ in texts]  # 返回零向量
        
        logger.debug(f"转换{len(texts)}个文本为BERT向量")
        
        result = []
        # 分批处理，避免内存溢出
        batch_size = 16
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 对批次进行编码
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)
            
            # 获取模型输出
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # 使用最后一层的[CLS]令牌作为句子表示
            sentence_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
            
            # 归一化
            for embedding in sentence_embeddings:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    normalized_embedding = embedding / norm
                else:
                    normalized_embedding = embedding
                result.append(normalized_embedding.tolist())
        
        return result
    
    def save(self, path: str) -> bool:
        """保存向量化器到指定路径。
        
        Args:
            path: 保存路径
            
        Returns:
            是否保存成功
        """
        try:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            logger.info(f"BERT向量化器已保存到{path}")
            return True
        except Exception as e:
            logger.error(f"保存BERT向量化器失败: {str(e)}")
            return False
    
    @classmethod
    def load(cls, path: str) -> "BERTVectorizer":
        """从指定路径加载向量化器。
        
        Args:
            path: 加载路径
            
        Returns:
            加载的向量化器实例
        """
        try:
            instance = cls.__new__(cls)
            instance.device = "cuda" if torch.cuda.is_available() else "cpu"
            instance.tokenizer = AutoTokenizer.from_pretrained(path)
            instance.model = AutoModel.from_pretrained(path).to(instance.device)
            instance.is_fitted = True
            logger.info(f"从{path}加载BERT向量化器")
            return instance
        except Exception as e:
            logger.error(f"加载BERT向量化器失败: {str(e)}")
            return cls()  # 返回新实例 