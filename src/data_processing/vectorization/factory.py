"""
向量化工厂模块，用于创建不同类型的向量化器。
"""

import os
import logging
from typing import Optional, Dict, Any

from .base import BaseVectorizer
from .tfidf_vectorizer import TfidfVectorizer
from .word2vec_vectorizer import Word2VecVectorizer
from .bert_vectorizer import BertVectorizer
from .bge_vectorizer import BgeVectorizer

logger = logging.getLogger(__name__)

class VectorizationFactory:
    """向量化工厂类，用于创建不同类型的向量化器。"""
    
    _vectorizers = {
        'tfidf': TfidfVectorizer,
        'word2vec': Word2VecVectorizer,
        'bert': BertVectorizer,
        'bge-m3': BgeVectorizer
    }
    
    @classmethod
    def create_vectorizer(cls, method: str = 'tfidf', **kwargs) -> BaseVectorizer:
        """创建向量化器。
        
        Args:
            method: 向量化方法，支持 'tfidf', 'word2vec', 'bert', 'bge-m3'
            **kwargs: 传递给向量化器的其他参数
            
        Returns:
            向量化器实例
            
        Raises:
            ValueError: 如果指定的向量化方法不支持
        """
        method = method.lower()
        if method not in cls._vectorizers:
            supported_methods = ", ".join(cls._vectorizers.keys())
            raise ValueError(f"不支持的向量化方法: {method}。支持的方法有: {supported_methods}")
        
        try:
            # 获取环境变量配置
            config = cls._get_config_from_env(method)
            
            # 合并传入的参数
            config.update(kwargs)
            
            # 创建向量化器
            vectorizer_class = cls._vectorizers[method]
            logger.info(f"创建向量化器: {method}")
            return vectorizer_class(**config)
            
        except Exception as e:
            logger.error(f"创建向量化器失败: {str(e)}")
            raise
    
    @staticmethod
    def _get_config_from_env(method: str) -> Dict[str, Any]:
        """从环境变量获取配置。
        
        Args:
            method: 向量化方法
            
        Returns:
            配置字典
        """
        config = {}
        
        # 通用配置
        config['cache_dir'] = os.getenv('VECTORIZATION_CACHE_DIR', './cache/vectorization')
        
        # 获取通用的模型名称（如果适用）
        default_model = os.getenv('EMBEDDING_MODEL', '')
        
        # 特定方法的配置
        if method == 'tfidf':
            config['max_features'] = int(os.getenv('TFIDF_MAX_FEATURES', '5000'))
            config['use_idf'] = os.getenv('TFIDF_USE_IDF', 'true').lower() == 'true'
            
        elif method == 'word2vec':
            config['vector_size'] = int(os.getenv('WORD2VEC_VECTOR_SIZE', '100'))
            config['window'] = int(os.getenv('WORD2VEC_WINDOW', '5'))
            config['min_count'] = int(os.getenv('WORD2VEC_MIN_COUNT', '1'))
            config['workers'] = int(os.getenv('WORD2VEC_WORKERS', '4'))
            config['pretrained_path'] = os.getenv('WORD2VEC_PRETRAINED_PATH', '')
            
        elif method == 'bert':
            # 优先使用BERT_MODEL_NAME，如果未设置则尝试使用通用EMBEDDING_MODEL
            bert_default = 'bert-base-uncased'
            if default_model and 'bert' in default_model.lower():
                bert_default = default_model
            
            config['model_name'] = os.getenv('BERT_MODEL_NAME', bert_default)
            config['max_length'] = int(os.getenv('BERT_MAX_LENGTH', '512'))
            config['device'] = os.getenv('EMBEDDING_DEVICE', os.getenv('BERT_DEVICE', 'cpu'))
            
        elif method == 'bge-m3':
            # 优先使用BGE_MODEL_NAME，如果未设置则尝试使用通用EMBEDDING_MODEL
            bge_default = 'BAAI/bge-m3'
            if default_model and 'bge' in default_model.lower():
                bge_default = default_model
            
            config['model_name'] = os.getenv('BGE_MODEL_NAME', bge_default)
            config['max_length'] = int(os.getenv('BGE_MAX_LENGTH', '512'))
            config['device'] = os.getenv('EMBEDDING_DEVICE', os.getenv('BGE_DEVICE', 'cpu'))
        
        return config 