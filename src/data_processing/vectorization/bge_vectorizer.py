"""
BGE-M3向量化器模块，使用BAAI/bge-m3模型将文本转换为向量。
"""

import logging
import numpy as np
from typing import List, Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

from .base import BaseVectorizer

logger = logging.getLogger(__name__)

class BgeVectorizer(BaseVectorizer):
    """BGE-M3向量化器，使用BAAI/bge-m3模型将文本转换为向量。"""
    
    def __init__(self, model_name: str = 'BAAI/bge-m3', max_length: int = 512, 
                 device: str = 'cpu', cache_dir: str = './cache/vectorization'):
        """初始化BGE-M3向量化器。
        
        Args:
            model_name: 模型名称或路径，默认为BAAI/bge-m3
            max_length: 最大序列长度
            device: 计算设备，'cpu'或'cuda'
            cache_dir: 缓存目录
        """
        super().__init__(cache_dir)
        self.model_name = model_name
        self.max_length = max_length
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        
        # 初始化模型和分词器
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载BGE-M3模型和分词器。"""
        try:
            logger.info(f"加载BGE-M3模型: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # 设置为评估模式
            logger.info(f"BGE-M3模型加载成功，使用设备: {self.device}")
        except Exception as e:
            logger.error(f"加载BGE-M3模型失败: {str(e)}")
            raise
    
    def _mean_pooling(self, model_output, attention_mask):
        """对模型输出进行平均池化。
        
        Args:
            model_output: 模型输出
            attention_mask: 注意力掩码
            
        Returns:
            池化后的向量
        """
        # 获取最后一层的隐藏状态
        token_embeddings = model_output.last_hidden_state
        
        # 扩展注意力掩码
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # 对掩码位置求和并除以掩码和
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def vectorize(self, text: str) -> np.ndarray:
        """将文本转换为向量。
        
        Args:
            text: 要向量化的文本
            
        Returns:
            文本的向量表示
        """
        try:
            # 确保模型已加载
            if self.model is None or self.tokenizer is None:
                self._load_model()
            
            # 为BGE-M3模型添加指令前缀（如果需要）
            # 注意：根据具体模型的使用要求，某些模型可能需要特定前缀
            instruction = "为这个句子生成表示以用于检索相关文章："
            if "bge-m3" in self.model_name.lower():
                if not text.startswith(instruction):
                    text = f"{instruction} {text}"
            
            # 对文本进行编码
            encoded_input = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # 将输入移动到指定设备
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            # 计算嵌入
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # 对输出进行池化
            sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            # 归一化向量（BGE-M3通常需要L2归一化）
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            
            # 转换为numpy数组
            return sentence_embeddings[0].cpu().numpy()
            
        except Exception as e:
            logger.error(f"BGE-M3向量化失败: {str(e)}")
            # 返回零向量
            return np.zeros(self.get_embedding_dim())
    
    def batch_vectorize(self, texts: List[str]) -> List[np.ndarray]:
        """批量将文本转换为向量。
        
        Args:
            texts: 要向量化的文本列表
            
        Returns:
            文本的向量表示列表
        """
        try:
            # 确保模型已加载
            if self.model is None or self.tokenizer is None:
                self._load_model()
            
            # 为BGE-M3模型添加指令前缀（如果需要）
            instruction = "为这个句子生成表示以用于检索相关文章："
            if "bge-m3" in self.model_name.lower():
                texts = [f"{instruction} {text}" if not text.startswith(instruction) else text for text in texts]
            
            # 对文本进行编码
            encoded_input = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # 将输入移动到指定设备
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            # 计算嵌入
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # 对输出进行池化
            sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            # 归一化向量（BGE-M3通常需要L2归一化）
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            
            # 转换为numpy数组列表
            return [embedding.cpu().numpy() for embedding in sentence_embeddings]
            
        except Exception as e:
            logger.error(f"BGE-M3批量向量化失败: {str(e)}")
            # 返回零向量列表
            return [np.zeros(self.get_embedding_dim()) for _ in texts]
    
    def get_embedding_dim(self) -> int:
        """获取嵌入维度。
        
        Returns:
            嵌入维度
        """
        try:
            config = AutoConfig.from_pretrained(self.model_name)
            return config.hidden_size
        except Exception as e:
            logger.error(f"获取嵌入维度失败: {str(e)}")
            # BGE-M3的默认维度是1024
            return 1024 