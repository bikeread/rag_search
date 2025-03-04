"""
BGE-M3向量化器实现。
"""

import os
import logging
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig

from .base import BaseVectorizer

logger = logging.getLogger(__name__)

class BGEVectorizer(BaseVectorizer):
    """基于BGE-M3的向量化器实现。"""
    
    def __init__(self, model_name: str = 'BAAI/bge-m3', max_length: int = 512, 
                 device: str = None, cache_dir: str = './cache/vectorization'):
        """初始化BGE-M3向量化器。
        
        Args:
            model_name: 模型名称或路径，默认为BAAI/bge-m3
            max_length: 最大序列长度
            device: 计算设备，默认为None（自动选择）
            cache_dir: 缓存目录
        """
        super().__init__(cache_dir)
        self.model_name = model_name
        self.max_length = max_length
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            logger.info(f"加载BGE-M3模型: {model_name}, 设备: {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()  # 设置为评估模式
            self.is_fitted = True
        except Exception as e:
            logger.error(f"加载BGE-M3模型失败: {str(e)}")
            self.tokenizer = None
            self.model = None
            self.is_fitted = False
    
    def fit(self, texts: List[str]) -> "BGEVectorizer":
        """BGE-M3模型不需要额外训练，此方法仅作兼容。
        
        Args:
            texts: 文本列表
            
        Returns:
            返回自身实例，方便链式调用
        """
        logger.info("BGE-M3向量化器无需训练")
        return self
    
    def _mean_pooling(self, model_output, attention_mask):
        """对模型输出进行平均池化。
        
        Args:
            model_output: 模型输出
            attention_mask: 注意力掩码
            
        Returns:
            池化后的向量
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def transform(self, texts: List[str]) -> List[List[float]]:
        """将文本列表转换为向量列表。
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表
        """
        if not self.is_fitted:
            logger.error("BGE-M3模型未正确加载，无法进行向量化")
            return [[0.0] * self.get_embedding_dim() for _ in texts]  # 返回零向量
        
        logger.debug(f"转换{len(texts)}个文本为BGE-M3向量")
        
        result = []
        # 分批处理，避免内存溢出
        batch_size = 16
        
        # 为BGE-M3模型添加指令前缀
        instruction = "为这个句子生成表示以用于检索相关文章："
        processed_texts = []
        for text in texts:
            if "bge-m3" in self.model_name.lower() and not text.startswith(instruction):
                processed_texts.append(f"{instruction} {text}")
            else:
                processed_texts.append(text)
        
        for i in range(0, len(processed_texts), batch_size):
            batch_texts = processed_texts[i:i+batch_size]
            
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
            
            # 平均池化获取句子嵌入
            sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            # L2归一化
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            
            # 转换为列表并添加到结果
            for embedding in sentence_embeddings:
                result.append(embedding.cpu().numpy().tolist())
        
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
            logger.info(f"BGE-M3向量化器已保存到{path}")
            return True
        except Exception as e:
            logger.error(f"保存BGE-M3向量化器失败: {str(e)}")
            return False
    
    @classmethod
    def load(cls, path: str) -> "BGEVectorizer":
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
            logger.info(f"从{path}加载BGE-M3向量化器")
            return instance
        except Exception as e:
            logger.error(f"加载BGE-M3向量化器失败: {str(e)}")
            return cls()  # 返回新实例