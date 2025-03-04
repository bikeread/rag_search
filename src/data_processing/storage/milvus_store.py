"""
Milvus vector store implementation.
"""

import os
import numpy as np
import hashlib
import logging
from typing import List, Optional, Set, Dict, Any, Tuple, Union
from pathlib import Path
import jieba
from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType
)
from sentence_transformers import SentenceTransformer

from langchain.embeddings.base import Embeddings
from langchain.schema import Document

from .base import VectorStoreBase, VectorStoreConfig
from ..vectorization.factory import VectorizationFactory
from ..processors.base import DocumentType
from src.rag.field_mapping import field_mapping_manager

logger = logging.getLogger(__name__)

class MilvusVectorStore(VectorStoreBase):
    """
    Milvus 向量存储。
    """
    
    def __init__(self, collection_name: str, embeddings: Any, host: str = "localhost", port: int = 19530):
        """
        初始化 Milvus 向量存储。

        Args:
            collection_name (str): 集合名称
            embeddings (Any): 嵌入模型
            host (str, optional): Milvus 服务器地址. 默认为 "localhost"
            port (int, optional): Milvus 服务器端口. 默认为 19530
        """
        self.logger = logging.getLogger(__name__)
        self.collection_name = collection_name
        
        # 初始化嵌入模型
        self.model = SentenceTransformer(embeddings)
        self.logger.info(f"创建嵌入模型:\n {embeddings}\n {self.model.device}，使用设备: {self.model.device}")
        
        # 连接到 Milvus 服务器
        connections.connect(
            alias="default",
            host=host,
            port=port
        )
        
        # 获取或创建集合
        self.collection = self._get_or_create_collection()
        
    def _connect(self):
        """连接 Milvus 服务器。"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            self.logger.info(f"Connected to Milvus server at {self.host}:{self.port}")
            
            # 检查集合是否存在
            if not utility.has_collection(self.collection_name):
                # 创建集合
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # BERT 模型的维度
                    FieldSchema(name="metadata", dtype=DataType.JSON)
                ]
                schema = CollectionSchema(fields=fields, description="Document collection")
                self.collection = Collection(self.collection_name, schema)
                
                # 创建索引
                index_params = {
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
                self.collection.create_index(field_name="embedding", index_params=index_params)
                self.logger.info(f"Created collection: {self.collection_name}")
            else:
                # 加载现有集合
                self.collection = Collection(self.collection_name)
            
            # 加载集合
            self.collection.load()
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Milvus: {e}")
            raise
            
    def search(self, query: str, top_k: int = None) -> List[Document]:
        """
        搜索与查询最相似的文档。

        Args:
            query (str): 查询字符串
            top_k (int, optional): 返回的最相似文档数量. 如果不指定，将使用环境变量TOP_K_RESULTS的值，默认为3.

        Returns:
            List[Document]: 相似文档列表
        """
        self.logger.info(f"开始搜索查询: '{query}'")
        
        # 如果没有指定top_k，从环境变量读取
        if top_k is None:
            top_k = int(os.getenv('TOP_K_RESULTS', '3'))
        
        # 获取搜索倍数
        search_multiplier = float(os.getenv('SEARCH_MULTIPLIER', '3.0'))
        expected_candidates = int(top_k * search_multiplier)
        self.logger.info(f"搜索配置 - TOP_K: {top_k} (from: {'parameter' if top_k is not None else 'env'}), SEARCH_MULTIPLIER: {search_multiplier}")
        self.logger.info(f"期望检索的候选数量: {expected_candidates}")
        
        # 生成查询向量
        query_vector = self.model.encode(query)
        self.logger.info(f"生成查询向量，维度: {len(query_vector)}")
        
        # 设置搜索参数
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10}
        }
        self.logger.info(f"搜索参数: {search_params}")
        
        # 执行搜索
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=expected_candidates,  # 使用扩大后的检索数量
            output_fields=["content", "metadata"]
        )
        
        documents = []
        if results:
            hits = results[0]  # 获取第一个查询的结果
            self.logger.info(f"实际检索到的候选数量: {len(hits)}")
            
            # 根据相似度得分筛选结果
            filtered_hits = hits[:top_k]  # 只保留top_k个结果
            self.logger.info(f"筛选后的最终结果数量: {len(filtered_hits)}")
            
            for hit in filtered_hits:
                try:
                    # 使用 to_dict() 方法获取完整的数据
                    hit_dict = hit.to_dict()
                    content = hit_dict['entity']['content']
                    metadata = hit_dict['entity'].get('metadata', {})
                    score = hit_dict['distance']  # 使用 distance 而不是 score
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            **metadata,
                            "score": score
                        }
                    )
                    documents.append(doc)
                except Exception as e:
                    self.logger.error(f"处理搜索结果时发生错误: {str(e)}")
                    self.logger.error(f"Hit 对象: {hit}")
                    self.logger.error(f"Hit 属性: {dir(hit)}")
                    continue
                    
        return documents
        
    def add_documents(self, documents: List[Union[str, Dict[str, Any], Document]]) -> bool:
        """
        添加文档。

        Args:
            documents (List[Union[str, Dict[str, Any], Document]]): 文档列表，可以是字符串、字典或Document对象

        Returns:
            bool: 是否添加成功
        """
        try:
            # 准备数据
            data_to_insert = []
            
            for doc in documents:
                try:
                    # 根据输入类型处理文档内容
                    if isinstance(doc, str):
                        content = doc
                        metadata = {}
                    elif isinstance(doc, dict):
                        content = doc.get("content", doc.get("page_content", ""))
                        metadata = doc.get("metadata", {})
                    elif isinstance(doc, Document):
                        content = doc.page_content
                        metadata = doc.metadata
                    else:
                        self.logger.warning(f"跳过不支持的文档类型: {type(doc)}")
                        continue

                    # 确保content不为空
                    if not content:
                        self.logger.warning("跳过空内容的文档")
                        continue

                    # 生成向量
                    embedding = self.model.encode(content)
                    self.logger.info(f"生成向量，维度: {len(embedding)}")
                    self.logger.debug(f"文档内容: {content[:100]}...")
                    
                    # 准备插入的数据
                    data = {
                        "content": content,
                        "embedding": embedding.tolist(),  # 转换为列表
                        "metadata": metadata
                    }
                    data_to_insert.append(data)
                    self.logger.debug(f"准备插入数据: {data.keys()}")
                except Exception as doc_error:
                    self.logger.error(f"处理单个文档时出错: {str(doc_error)}")
                    continue
            
            if not data_to_insert:
                self.logger.warning("没有可插入的有效数据")
                return False

            # 插入数据
            self.logger.info(f"开始插入 {len(data_to_insert)} 条数据")
            insert_result = self.collection.insert(data_to_insert)
            self.logger.info(f"插入结果: {insert_result}")
            
            # 创建索引
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.logger.info(f"创建索引，参数: {index_params}")
            self.collection.create_index(field_name="embedding", index_params=index_params)
            
            # 加载集合
            self.collection.load()
            self.logger.info(f"成功添加 {len(data_to_insert)} 个文档")
            
            return True
            
        except Exception as e:
            self.logger.error(f"添加文档失败: {str(e)}")
            return False
            
    def delete_documents(self, ids: List[str]) -> bool:
        """
        删除文档。

        Args:
            ids (List[str]): 文档ID列表

        Returns:
            bool: 是否删除成功
        """
        try:
            expr = f"id in {ids}"
            self.collection.delete(expr)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete documents: {e}")
            return False
    
    def _compute_hash(self, text: str) -> str:
        """计算文本哈希值"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _parse_document_content(self, doc: str) -> Dict[str, str]:
        """解析文档内容"""
        try:
            self.logger.debug(f"解析文档内容: {doc[:100]}...")
            
            # 处理按行分隔的格式（键值对形式）
            if "\n" in doc:
                parts = {}
                for line in doc.split("\n"):
                    if ": " in line:
                        key, value = line.split(": ", 1)
                        std_field = field_mapping_manager.get_standard_field_name(key.strip())
                        parts[std_field] = value.strip()
                if parts:  # 如果成功解析了一些键值对，返回结果
                    self.logger.debug(f"按键值对格式成功解析文档: {parts}")
                    return parts
                
                # 如果按键值对解析失败，尝试按Markdown表格格式解析
                lines = [line.strip() for line in doc.split("\n") if line.strip()]
                if len(lines) >= 3 and all("|" in line for line in lines):
                    self.logger.debug("尝试按Markdown表格格式解析")
                    try:
                        # 尝试解析Markdown表格
                        header_line = lines[0]
                        separator_line = lines[1]
                        data_line = lines[2]
                        
                        # 解析表头
                        headers = [h.strip() for h in header_line.split("|") if h.strip()]
                        self.logger.debug(f"表头: {headers}")
                        
                        # 解析数据行
                        data_values = [d.strip() for d in data_line.split("|") if d.strip()]
                        self.logger.debug(f"数据值: {data_values}")
                        
                        # 确保headers和data_values长度一致
                        if len(headers) == len(data_values):
                            parts = {}
                            for i, header in enumerate(headers):
                                std_field = field_mapping_manager.get_standard_field_name(header)
                                parts[std_field] = data_values[i]
                            self.logger.debug(f"按表格格式成功解析文档: {parts}")
                            return parts
                        else:
                            self.logger.warning(f"表头和数据行长度不匹配: 表头长度={len(headers)}, 数据行长度={len(data_values)}")
                    except Exception as table_error:
                        self.logger.warning(f"解析表格格式失败: {str(table_error)}")
            
            # 处理按 | 分隔的格式
            elif " | " in doc:
                parts = {}
                for item in doc.split(" | "):
                    if ": " in item:
                        key, value = item.split(": ", 1)
                        std_field = field_mapping_manager.get_standard_field_name(key.strip())
                        parts[std_field] = value.strip()
                if parts:
                    self.logger.debug(f"按pipe分隔格式成功解析文档: {parts}")
                    return parts
            
            self.logger.warning(f"无法解析文档内容: {doc[:100]}...")
            return {}
            
        except Exception as e:
            self.logger.warning(f"Error parsing document content: {str(e)}")
            return {}
    
    def _check_category_match(self, query: str, doc_category: str) -> float:
        """检查类别匹配度"""
        try:
            words = list(jieba.cut(query))
            max_similarity = 0.0
            
            for word in words:
                word_embedding = self.model.encode(word)
                category_embedding = self.model.encode(doc_category)
                similarity = self.compute_similarity(word_embedding, category_embedding)
                max_similarity = max(max_similarity, similarity)
            
            return max_similarity
            
        except Exception as e:
            self.logger.warning(f"Error checking category match: {str(e)}")
            return 0.0
    
    def save(self) -> None:
        """保存向量存储状态"""
        # Milvus自动保存，无需实现
        pass
    
    def load(self) -> None:
        """加载向量存储状态"""
        # 重新加载集合
        self.collection.load()
    
    def get_embedding(self, text: str) -> np.ndarray:
        """获取文本的向量表示"""
        if not text:
            return np.zeros(self.model.get_sentence_embedding_dimension())
        return self.model.encode(text)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """计算两个向量之间的相似度"""
        try:
            if embedding1.ndim == 1:
                embedding1 = embedding1.reshape(1, -1)
            if embedding2.ndim == 1:
                embedding2 = embedding2.reshape(1, -1)
            
            norm_emb1 = embedding1 / (np.linalg.norm(embedding1, axis=1, keepdims=True) + 1e-8)
            norm_emb2 = embedding2 / (np.linalg.norm(embedding2, axis=1, keepdims=True) + 1e-8)
            
            similarity = np.dot(norm_emb1, norm_emb2.T).flatten()[0]
            return max(0.0, min(1.0, float(similarity)))
            
        except Exception as e:
            self.logger.error(f"Error computing similarity: {str(e)}")
            return 0.0
    
    def _get_or_create_collection(self) -> Collection:
        """
        获取或创建 Milvus 集合。

        Returns:
            Collection: Milvus 集合对象
        """
        try:
            # 定义集合的 schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            schema = CollectionSchema(
                fields=fields,
                description="Document collection"
            )

            # 检查集合是否存在
            if utility.has_collection(self.collection_name):
                collection = Collection(self.collection_name)
                self.logger.info(f"Using existing collection: {self.collection_name}")
            else:
                # 创建新集合
                collection = Collection(
                    name=self.collection_name,
                    schema=schema,
                    using='default'
                )
                # 创建索引
                index_params = {
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
                collection.create_index(field_name="embedding", index_params=index_params)
                self.logger.info(f"Created new collection: {self.collection_name}")

            # 加载集合
            collection.load()
            return collection

        except Exception as e:
            self.logger.error(f"Failed to get or create collection: {e}")
            raise 