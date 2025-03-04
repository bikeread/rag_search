"""
FAISS vector store implementation.
"""

import os
import faiss
import numpy as np
import hashlib
import logging
from typing import List, Optional, Set, Dict, Any
from pathlib import Path
from sentence_transformers import SentenceTransformer
import jieba
import pickle

from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

from src.data_processing.storage.base import VectorStoreBase, VectorStoreConfig
from src.data_processing.vectorization.factory import VectorizationFactory
from src.data_processing.processors.base import DocumentType
from src.rag.field_mapping import field_mapping_manager

logger = logging.getLogger(__name__)

class FAISSVectorStore(VectorStoreBase):
    """FAISS向量存储实现"""
    
    def __init__(self, config, embeddings):
        """初始化FAISS向量存储
        
        Args:
            config: 向量存储配置
            embeddings: 嵌入模型
        """
        logger.info("初始化FAISS向量存储")
        self.embedding_model = embeddings
        self.config = config  # 保存配置对象
        self.index = None
        self.docstore = {}
        self.index_to_docstore_id = {}
        
        # 如果提供了路径，尝试加载现有索引
        if config.store_path:
            self.load_local(config.store_path)
    
    def add_documents(self, documents: List[Document]) -> bool:
        """添加文档到向量存储
        
        Args:
            documents: 文档列表
            
        Returns:
            bool: 是否成功添加
        """
        try:
            logger.info(f"添加 {len(documents)} 个文档到FAISS")
            
            # 提取文档内容和元数据
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # 创建或更新FAISS索引
            if self.index is None:
                # 获取文档的嵌入向量
                embeddings = self.embedding_model.embed_documents(texts)
                
                # 创建FAISS索引
                dimension = len(embeddings[0])
                self.index = faiss.IndexFlatL2(dimension)
                
                # 添加向量到索引
                faiss.normalize_L2(np.array(embeddings).astype('float32'))
                self.index.add(np.array(embeddings).astype('float32'))
                
                # 保存文档到文档存储
                for i, (text, metadata) in enumerate(zip(texts, metadatas)):
                    doc_id = metadata.get("product_id", str(i))
                    self.docstore[doc_id] = Document(page_content=text, metadata=metadata)
                    self.index_to_docstore_id[i] = doc_id
                
                logger.info(f"创建了新的FAISS索引，维度: {dimension}")
            else:
                # 获取文档的嵌入向量
                embeddings = self.embedding_model.embed_documents(texts)
                
                # 添加向量到索引
                faiss.normalize_L2(np.array(embeddings).astype('float32'))
                start_index = len(self.index_to_docstore_id)
                self.index.add(np.array(embeddings).astype('float32'))
                
                # 保存文档到文档存储
                for i, (text, metadata) in enumerate(zip(texts, metadatas)):
                    doc_id = metadata.get("product_id", str(start_index + i))
                    self.docstore[doc_id] = Document(page_content=text, metadata=metadata)
                    self.index_to_docstore_id[start_index + i] = doc_id
                
                logger.info(f"添加了 {len(documents)} 个文档到现有FAISS索引")
            
            return True
            
        except Exception as e:
            logger.error(f"添加文档到FAISS失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def search(self, query: str, k: int = 4, filter_threshold: bool = True) -> List[Document]:
        """搜索向量存储
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filter_threshold: 是否应用相似度阈值过滤
            
        Returns:
            相关文档列表及其相似度分数
        """
        try:
            logger.info(f"执行FAISS搜索: '{query}'，返回结果数量: {k}")
            
            # 向量化查询
            query_embedding = self.embedding_model.embed_query(query)
            
            # 执行搜索
            if self.index is None:
                logger.warning("FAISS索引为空，无法执行搜索")
                return []
                
            logger.info(f"FAISS索引大小: {self.index.ntotal}")
            
            doc_scores, doc_indices = self.index.search(
                np.array([query_embedding], dtype=np.float32), 
                k=k
            )
            
            logger.info(f"FAISS搜索返回 {len(doc_indices[0])} 个结果")
            logger.info(f"文档索引: {doc_indices[0]}")
            logger.info(f"文档分数: {doc_scores[0]}")
            
            # 检查索引映射和文档存储
            if not self.index_to_docstore_id:
                logger.warning("索引映射为空，无法获取文档ID")
                # 尝试直接使用索引作为文档ID
                for i, doc_index in enumerate(doc_indices[0]):
                    if doc_index != -1:
                        self.index_to_docstore_id[doc_index] = str(doc_index)
            
            if not self.docstore:
                logger.warning("文档存储为空，无法获取文档内容")
                return []
            
            results = []
            min_score = self.config.min_similarity_score if hasattr(self, 'config') else float(os.getenv("MIN_SIMILARITY_SCORE", "0.0"))
            logger.info(f"使用最小相似度分数: {min_score}")
            
            for i, (doc_index, score) in enumerate(zip(doc_indices[0], doc_scores[0])):
                if doc_index == -1:  # FAISS返回-1表示没有更多结果
                    logger.debug(f"跳过索引 {i}，FAISS返回-1")
                    continue
                    
                # 尝试获取文档ID
                doc_id = self.index_to_docstore_id.get(doc_index)
                if doc_id is None:
                    logger.warning(f"跳过索引 {doc_index}，未找到对应的文档ID")
                    # 尝试直接使用索引作为文档ID
                    doc_id = str(doc_index)
                    
                # 尝试获取文档
                document = self.docstore.get(doc_id)
                if document is None:
                    logger.warning(f"跳过文档ID {doc_id}，未找到对应的文档")
                    continue
                
                # 计算余弦相似度 (FAISS默认返回的是L2距离)
                # 注意：FAISS返回的L2距离可能很大，导致计算出的余弦相似度为负值
                # 使用更合适的转换方式，确保分数在0-1之间
                # 使用指数衰减函数将L2距离转换为相似度分数
                cosine_score = np.exp(-score / 10)  # 使用指数衰减，确保分数在0-1之间
                
                logger.info(f"文档ID: {doc_id}, L2距离: {score:.4f}, 转换后的相似度分数: {cosine_score:.4f}")
                
                # 如果启用了阈值过滤并且分数低于阈值，跳过此文档
                if filter_threshold and cosine_score < min_score:
                    logger.info(f"文档分数 {cosine_score:.4f} 低于阈值 {min_score}，已过滤")
                    continue
                    
                # 将相似度分数添加到文档的元数据中
                document.metadata["score"] = float(cosine_score)
                results.append(document)
                
            logger.info(f"FAISS搜索找到 {len(results)}/{len(doc_indices[0])} 个文档满足条件")
            return results
            
        except Exception as e:
            logger.error(f"FAISS搜索异常: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def save_local(self) -> bool:
        """保存索引到本地
        
        Returns:
            bool: 是否成功保存
        """
        try:
            if not self.index:
                logger.warning("没有索引可保存")
                return False
                
            # 获取保存路径
            index_path = self.config.connection_args.get("vector_db_path", self.config.store_path)
            if not index_path:
                logger.warning("未指定索引保存路径")
                return False
                
            # 确保目录存在
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            
            # 保存FAISS索引
            faiss_path = f"{index_path}.faiss"
            pkl_path = f"{index_path}.pkl"
            
            faiss.write_index(self.index, faiss_path)
            with open(pkl_path, "wb") as f:
                pickle.dump((self.docstore, self.index_to_docstore_id), f)
                
            logger.info(f"成功保存FAISS索引到 {index_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存FAISS索引失败: {str(e)}")
            return False
            
    def load_local(self, index_path: str) -> bool:
        """从本地加载索引
        
        Args:
            index_path: 索引路径
            
        Returns:
            bool: 是否成功加载
        """
        try:
            if not index_path:
                logger.warning("未指定索引加载路径")
                return False
                
            # 加载FAISS索引
            faiss_path = f"{index_path}.faiss"
            pkl_path = f"{index_path}.pkl"
            
            if not os.path.exists(faiss_path) or not os.path.exists(pkl_path):
                logger.warning(f"索引文件不存在: {faiss_path} 或 {pkl_path}")
                return False
                
            # 加载FAISS索引
            self.index = faiss.read_index(faiss_path)
            logger.info(f"加载FAISS索引，大小: {self.index.ntotal}")
            
            # 加载文档存储和映射关系
            with open(pkl_path, "rb") as f:
                self.docstore, self.index_to_docstore_id = pickle.load(f)
                
            logger.info(f"加载文档存储，大小: {len(self.docstore)}")
            logger.info(f"加载索引映射，大小: {len(self.index_to_docstore_id)}")
            
            # 打印一些映射信息，用于调试
            if len(self.index_to_docstore_id) > 0:
                sample_keys = list(self.index_to_docstore_id.keys())[:3]
                logger.info(f"索引映射示例: {sample_keys} -> {[self.index_to_docstore_id[k] for k in sample_keys]}")
                
                sample_doc_ids = list(self.docstore.keys())[:3]
                logger.info(f"文档存储示例: {sample_doc_ids}")
            
            logger.info(f"成功从 {index_path} 加载FAISS索引")
            return True
            
        except Exception as e:
            logger.error(f"加载FAISS索引失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    def clear(self) -> bool:
        """清空向量存储
        
        Returns:
            是否成功清空
        """
        try:
            # 重置索引和文档存储
            self.index = None
            self.docstore = {}
            self.index_to_docstore_id = {}
            
            # 删除本地索引文件
            if self.config.index_path:
                faiss_path = f"{self.config.index_path}.faiss"
                pkl_path = f"{self.config.index_path}.pkl"
                
                if os.path.exists(faiss_path):
                    os.remove(faiss_path)
                    
                if os.path.exists(pkl_path):
                    os.remove(pkl_path)
                    
            logger.info("FAISS向量存储已清空")
            return True
            
        except Exception as e:
            logger.error(f"清空FAISS向量存储失败: {str(e)}")
            return False
            
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息
        
        Returns:
            包含统计信息的字典，如文档数量、向量维度等
        """
        if self.index is None:
            return {"status": "未初始化", "doc_count": 0}
            
        return {
            "status": "已加载", 
            "doc_count": self.index.ntotal,
            "dim": self.index.d if hasattr(self.index, 'd') else "未知",
            "is_trained": getattr(self.index, "is_trained", True)
        } 