"""
Retrieval-Augmented Generation (RAG) implementation.
"""

import logging
import os
import re
import time
from typing import List, Dict, Any, Optional, Tuple

from langchain.llms.base import BaseLLM
from langchain.schema import BaseRetriever
from langchain.schema import BaseRetriever as LCBaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from config.config import (
    MIN_SIMILARITY_SCORE,
    TOP_K_RESULTS,
    SEARCH_MULTIPLIER,
)
from src.chains.processors.base import QueryProcessor
from src.data_processing.processors.base import ProcessorConfig
from src.data_processing.processors.document_processor import DocumentProcessor
from src.data_processing.storage.base import VectorStoreBase
from src.utils.logging_utils import log_args_and_result  # 导入日志装饰器

# 导入默认处理器

# 配置日志
logger = logging.getLogger(__name__)

# 全局定义SerializableRetriever类
class SerializableRetriever(LCBaseRetriever):
    """可序列化的检索器包装类"""
    
    def __init__(self, retriever, **kwargs):
        """初始化
        
        Args:
            retriever: 原始检索器对象
        """
        super().__init__(**kwargs)
        # 存储检索器引用
        self._retriever = retriever
        # 存储检索器信息用于调试
        self._retriever_info = {
            "type": type(retriever).__name__,
            "id": id(retriever)
        }
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """调用原始检索器的方法"""
        try:
            # 检查检索器引用是否存在
            if self._retriever is None:
                logger.error("检索器引用丢失")
                return []
                
            # 尝试调用检索器
            if hasattr(self._retriever, 'invoke'):
                return self._retriever.invoke(query)
            elif hasattr(self._retriever, 'get_relevant_documents'):
                return self._retriever.get_relevant_documents(query)
            else:
                return self._retriever(query)
                
        except Exception as e:
            logger.error(f"检索器调用失败: {str(e)}")
            return []
            
    def __getstate__(self):
        """自定义序列化行为"""
        state = self.__dict__.copy()
        # 移除不可序列化的检索器引用
        state['_retriever'] = None
        return state
        
    def __setstate__(self, state):
        """自定义反序列化行为"""
        self.__dict__.update(state)

class VectorStoreRetriever(BaseRetriever, BaseModel):
    """向量存储检索器，提供语义检索功能"""

    vectorstore: VectorStoreBase = Field(description="Vector store for document retrieval")
    semantic_threshold: float = Field(default=MIN_SIMILARITY_SCORE, description="Semantic similarity threshold")
    top_k: int = Field(default=TOP_K_RESULTS, description="Number of top results to return")
    search_multiplier: int = Field(default=SEARCH_MULTIPLIER, description="Multiplier for search candidates")

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs
    ) -> List[Document]:
        """获取相关文档
        
        Args:
            query: 查询文本
            run_manager: 回调管理器
            
        Returns:
            List[Document]: 相关文档列表
        """
        try:
            logger.info(f"Getting relevant documents for query: '{query}'")
            
            # 获取初始文档集
            # 使用self.top_k作为搜索数量
            docs = self.vectorstore.search(
                query=query, 
                k=self.top_k * self.search_multiplier
            )
            
            # 过滤低于语义阈值的文档
            filtered_docs = []
            for doc in docs:
                if "score" in doc.metadata and doc.metadata["score"] >= self.semantic_threshold:
                    filtered_docs.append(doc)
                elif "score" not in doc.metadata:
                    # 如果没有分数，保留文档
                    filtered_docs.append(doc)
            
            # 如果过滤后没有文档，但原始结果有文档，返回分数最高的那个
            if not filtered_docs and docs:
                docs.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)
                filtered_docs = [docs[0]]
            
            # 返回前top_k个结果
            return filtered_docs[:self.top_k]
        except Exception as e:
            logger.error(f"Error in retriever: {str(e)}", exc_info=True)
            return []


class QueryOptimizer:
    """查询优化器，提供各种查询优化方法以提高检索命中率。"""
    
    def __init__(self, llm=None):
        """初始化查询优化器。
        
        Args:
            llm: 可选的语言模型，用于查询改写和扩展
        """
        self.llm = llm
    
    def decompose_query(self, query: str) -> List[str]:
        """将复杂查询拆分为多个简单子查询。
        
        Args:
            query: 原始查询
            
        Returns:
            子查询列表
        """
        if not self.llm:
            logger.warning("未提供LLM，无法执行查询拆分")
            return [query]
            
        try:
            prompt = f"""
            请将以下用户查询拆分成2-3个更基础的子查询，以便于在文档中检索信息：
            原始查询：{query}
            
            请直接输出拆分后的子查询，每行一个，不要添加额外说明。
            """
            response = self.llm.invoke(prompt)
            
            # 过滤思考过程，移除<think>...</think>之间的内容
            if "<think>" in response and "</think>" in response:
                think_start = response.find("<think>")
                think_end = response.find("</think>") + len("</think>")
                response = response[:think_start] + response[think_end:]
            
            # 清理并过滤结果
            sub_queries = []
            for line in response.split('\n'):
                line = line.strip()
                # 跳过空行和包含<think>标记的行
                if not line or "<think>" in line or "</think>" in line:
                    continue
                # 移除序号前缀（如"1. "）
                if re.match(r'^\d+[\.\)]\s+', line):
                    line = re.sub(r'^\d+[\.\)]\s+', '', line)
                if line:
                    sub_queries.append(line)
            
            if not sub_queries:
                logger.warning("查询拆分结果为空或格式不正确，回退到原始查询")
                return [query]
                
            logger.info(f"查询拆分结果: {sub_queries}")
            return sub_queries
        except Exception as e:
            logger.error(f"查询拆分失败: {str(e)}")
            return [query]
    
    def expand_query(self, query: str) -> List[str]:
        """生成查询的多种变体和扩展形式。
        
        Args:
            query: 原始查询
            
        Returns:
            查询变体列表
        """
        if not self.llm:
            logger.warning("未提供LLM，无法执行查询扩展")
            return [query]
            
        try:
            prompt = f"""
            请为以下查询生成3-4个语义相近但表达不同的变体形式，包括同义词替换、概念扩展等：
            原始查询：{query}
            
            请直接输出变体，每行一个，不要添加额外说明。
            """
            response = self.llm.invoke(prompt)
            
            # 过滤思考过程，移除<think>...</think>之间的内容
            if "<think>" in response and "</think>" in response:
                think_start = response.find("<think>")
                think_end = response.find("</think>") + len("</think>")
                response = response[:think_start] + response[think_end:]
            
            # 清理并过滤结果
            variants = []
            for line in response.split('\n'):
                line = line.strip()
                # 跳过空行和包含<think>标记的行
                if not line or "<think>" in line or "</think>" in line:
                    continue
                # 移除序号前缀（如"1. "）
                if re.match(r'^\d+[\.\)]\s+', line):
                    line = re.sub(r'^\d+[\.\)]\s+', '', line)
                if line:
                    variants.append(line)
            
            if not variants:
                logger.warning("查询扩展结果为空或格式不正确，回退到原始查询")
                return [query]
                
            logger.info(f"查询扩展结果: {variants}")
            return variants
        except Exception as e:
            logger.error(f"查询扩展失败: {str(e)}")
            return [query]
    
    def extract_keywords(self, query: str) -> List[str]:
        """从查询中提取关键词。
        
        Args:
            query: 原始查询
            
        Returns:
            关键词列表
        """
        if not self.llm:
            # 简单分词策略，如果没有提供LLM
            import jieba
            return [w for w in jieba.cut(query) if len(w) > 1]
            
        try:
            prompt = f"""
            请从以下用户查询中提取3-5个最重要的关键词，这些关键词应该能够用于检索相关文档：
            用户查询：{query}
            
            请直接输出关键词，每行一个，不要添加额外说明。
            """
            response = self.llm.invoke(prompt)
            
            # 过滤思考过程，移除<think>...</think>之间的内容
            if "<think>" in response and "</think>" in response:
                think_start = response.find("<think>")
                think_end = response.find("</think>") + len("</think>")
                response = response[:think_start] + response[think_end:]
            
            # 清理并过滤结果
            keywords = []
            for line in response.split('\n'):
                line = line.strip()
                # 跳过空行和包含<think>标记的行
                if not line or "<think>" in line or "</think>" in line:
                    continue
                # 移除序号前缀（如"1. "）
                if re.match(r'^\d+[\.\)]\s+', line):
                    line = re.sub(r'^\d+[\.\)]\s+', '', line)
                if line:
                    keywords.append(line)
            
            if not keywords:
                logger.warning("关键词提取结果为空或格式不正确，回退到简单分词")
                import jieba
                return [w for w in jieba.cut(query) if len(w) > 1]
                
            logger.info(f"关键词提取结果: {keywords}")
            return keywords
        except Exception as e:
            logger.error(f"关键词提取失败: {str(e)}")
            import jieba
            return [w for w in jieba.cut(query) if len(w) > 1]


class RAGRetriever:
    """
    RAG 检索器。
    """
    
    def __init__(
        self,
        vector_store: VectorStoreBase,
        llm: BaseLLM,
        query_enhancement_enabled: bool = True,
        chain_type: str = "qa"
    ):
        """
        初始化RAG检索器。

        Args:
            vector_store (VectorStoreBase): 向量存储
            llm (BaseLLM): 语言模型
            query_enhancement_enabled (bool, optional): 是否启用查询增强. 默认为 True.
            chain_type (str, optional): 链类型. 默认为 "qa".
        """
        self.logger = logging.getLogger(__name__)
        self.vector_store = vector_store
        self.llm = llm
        self.query_enhancement_enabled = query_enhancement_enabled
        self.chain_type = chain_type
        
        self.logger.info("Initializing RAG retriever")

    def _get_processor(self, query: str) -> QueryProcessor:
        """
        根据查询内容获取合适的查询处理器。
        
        Args:
            query (str): 用户查询
            
        Returns:
            QueryProcessor: 查询处理器
        """
        from src.chains.processors.factory import QueryProcessorFactory
        return QueryProcessorFactory.get_processor_for_query(query)

    @log_args_and_result
    def _get_relevant_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        获取与查询相关的文档。
        
        Args:
            query (str): 用户查询
            
        Returns:
            List[Dict[str, Any]]: 相关文档列表
        """
        return self.vector_store.search(query)
            
    @log_args_and_result
    def generate_answer(
        self, 
        query: str, 
        chat_history: Optional[List[Tuple[str, str]]] = None,
        chain_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成答案。
        
        Args:
            query (str): 用户查询
            chat_history (Optional[List[Tuple[str, str]]], optional): 聊天历史. 默认为 None.
            chain_type (Optional[str], optional): 链类型，如果提供则覆盖初始化时的设置. 默认为 None.
            **kwargs: 额外的参数
            
        Returns:
            Dict[str, Any]: 包含答案和来源的字典
        """
        start_time = time.time()
        # 使用传入的 chain_type 或默认值
        current_chain_type = chain_type or self.chain_type
        self.logger.info(f"Set chain type to: {current_chain_type}")
        self.logger.info(f"Generating answer for query: {query} using chain_type: {current_chain_type}")
        
        try:
            # 获取相关文档
            documents = self._get_relevant_documents(query)
            self.logger.info(f"找到 {len(documents)} 个相关文档")
            # 打印文档内容
            self.logger.info("检索到的文档内容:")
            for i, doc in enumerate(documents):
                self.logger.info(f"文档 {i+1}:")
                self.logger.info(f"内容: {doc.page_content[:200]}...")  # 只打印前200个字符
                self.logger.info(f"元数据: {doc.metadata}")
            # 获取适合的处理器
            processor = self._get_processor(query)
            self.logger.info(f"Using specialized processor: {processor.__class__.__name__}")
            
            # 处理查询
            result = processor.process(
                query=query,
                documents=documents,
                chat_history=chat_history,
                chain_type=current_chain_type,
                **kwargs
            )
            
            # 记录处理时间
            end_time = time.time()
            self.logger.info(f"成功耗时: {end_time - start_time:.2f}秒")
            
            return result
            
        except Exception as e:
            # 记录错误和处理时间
            end_time = time.time()
            self.logger.error(f"Error generating answer: {e}")
            self.logger.error(f"失败耗时: {end_time - start_time:.2f}秒")
            
            # 返回错误信息
            return {
                "answer": "抱歉，生成答案时发生错误。请稍后重试。",
                "sources": []
            }

    def add_documents(self, documents: List[bytes], mime_type: str = "text/plain", doc_type: str = "auto") -> None:
        """Add documents to the knowledge base.
        
        Args:
            documents: List of document contents in bytes
            mime_type: MIME type of the documents
            doc_type: Document type for text splitting (auto, short_text, normal_text, etc.)
        """
        logger.info(f"Adding {len(documents)} documents of type {mime_type} (doc_type: {doc_type})")

        # 创建文档处理器配置
        config = ProcessorConfig(doc_type=doc_type)
        
        # 创建文档处理器
        processor = DocumentProcessor(config)

        # 处理Excel文件的MIME类型列表
        EXCEL_MIME_TYPES = [
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel.sheet.macroEnabled.12"
        ]
        
        # 根据 MIME 类型确定文档类型
        if mime_type in EXCEL_MIME_TYPES:
            logger.debug("检测到Excel文件，使用表格处理器")
            from src.data_processing.processors.table_processor import TableProcessor, TableProcessorConfig
            # 创建表格处理器专用配置
            table_config = TableProcessorConfig(
                doc_type=doc_type,
                max_rows_per_chunk=int(os.getenv('TABLE_MAX_ROWS_PER_CHUNK', '5')),
                preserve_empty_cells=os.getenv('TABLE_PRESERVE_EMPTY_CELLS', 'false').lower() == 'true',
                enable_vectorization=os.getenv('ENABLE_VECTORIZATION', 'false').lower() == 'true',
                vectorization_method=os.getenv('VECTORIZATION_METHOD', 'tfidf')
            )
            processor = TableProcessor(table_config)

        # Process each document
        processed_docs = []
        for doc_content in documents:
            try:
                # Load and process document
                loaded_docs = processor.process_file(doc_content, "document.tmp", mime_type)
                processed_docs.extend(loaded_docs)
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                continue

        if not processed_docs:
            logger.warning("No documents were successfully processed")
            return

        # Add to vector store
        texts = [doc.page_content for doc in processed_docs]
        self.vector_store.add_documents(texts)
        logger.info(f"Successfully added {len(processed_docs)} processed documents")

    def get_relevant_documents_enhanced(
        self, query: str, k: int = None, **kwargs
    ) -> List[Document]:
        """增强版检索方法，首次直接进行语义拆分和扩展，无需轮询
        
        Args:
            query: 用户查询
            k: 返回的文档数量上限
            
        Returns:
            相关文档列表
        """
        k = k or self.config.top_k_results
        logger.info(f"开始增强检索过程，查询: '{query}'")
        
        # 存储所有检索到的文档
        all_docs = []
        
        # 同时准备查询分解和查询扩展的列表
        all_queries = []
        
        # 1. 首先收集所有可能的查询变体
        if self.config.query_decomposition_enabled:
            # 分解查询
            sub_queries = self.query_optimizer.decompose_query(query)
            logger.info(f"查询分解结果: {len(sub_queries)}个子查询")
            all_queries.extend(sub_queries)
        
        if self.config.query_expansion_enabled:
            # 查询扩展
            expanded_queries = self.query_optimizer.expand_query(query)
            logger.info(f"查询扩展结果: {len(expanded_queries)}个扩展查询")
            all_queries.extend(expanded_queries)
        
        # 确保查询列表不为空
        if not all_queries:
            logger.warning("查询优化未产生任何子查询或扩展查询，将直接返回空结果")
            return []
        
        # 2. 一次性执行所有收集到的查询，直到找到足够的文档或者所有查询都尝试过
        logger.info(f"执行{len(all_queries)}个优化查询")
        
        for query_variant in all_queries:
            logger.info(f"执行查询: '{query_variant}'")
            
            try:
                # 使用invoke方法获取文档
                variant_docs = self.retriever.invoke(query_variant)
                
                # 过滤掉已经在结果中的文档
                new_docs = []
                for doc in variant_docs:
                    if not any(self._doc_exists_in_list(doc, existing_doc) for existing_doc in all_docs):
                        new_docs.append(doc)
                
                logger.info(f"查询 '{query_variant}' 新增了 {len(new_docs)} 个文档")
                all_docs.extend(new_docs)
                
                if len(all_docs) >= k:
                    logger.info(f"已找到足够的文档 ({len(all_docs)}), 停止查询")
                    break
            except Exception as e:
                logger.warning(f"查询 '{query_variant}' 失败: {str(e)}")
        
        # 3. 最终处理
        if all_docs:
            # 对结果进行排序，确保最相关的文档排在前面
            all_docs.sort(key=lambda x: x.metadata.get("score", 0), reverse=True)
            
            # 打印结果信息
            logger.info(f"最终找到 {len(all_docs)} 个文档")
            for i, doc in enumerate(all_docs[:min(len(all_docs), 3)]):
                score = doc.metadata.get("score", "未知")
                if isinstance(score, (int, float)):
                    score_str = f"{score:.4f}"
                else:
                    score_str = str(score)
                logger.info(f"  结果 #{i+1}: 相似度={score_str}, 内容={doc.page_content[:50]}...")
        else:
            logger.warning("未找到任何相关文档！")
            logger.info(f"请检查：1.查询 '{query}' 是否与文档内容相关 2.相似度阈值设置是否过高 3.向量存储是否包含足够数据")
            
        # 返回前k个文档
        return all_docs[:k]

    def _get_doc_id(self, doc: Document) -> str:
        """获取文档的唯一标识。"""
        # 尝试从元数据中获取ID
        if 'id' in doc.metadata:
            return str(doc.metadata['id'])
        if 'source' in doc.metadata:
            return str(doc.metadata['source'])
        # 如果没有ID，使用内容的hash
        return str(hash(doc.page_content))

    def get_relevant_documents(
        self, query: str, *, callbacks: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """获取与查询相关的文档
        
        Args:
            query: 用户查询
            callbacks: 回调管理器
            
        Returns:
            相关文档列表
        """
        logger.info(f"Getting relevant documents for query: '{query}'")
        
        # 添加调试日志：检查向量存储是否有数据
        try:
            logger.info(f"向量存储状态检查：数据量统计 = {self.vectorstore.get_stats()}")
        except Exception as e:
            logger.warning(f"无法获取向量存储统计数据: {str(e)}")
        
        # 记录查询配置
        logger.info(f"查询配置: top_k={self.top_k}, threshold={self.semantic_threshold}")
        
        # 在查询前记录查询词的向量
        if hasattr(self.vectorstore, "_get_embedding"):
            try:
                emb = self.vectorstore._get_embedding(query)
                logger.info(f"查询词向量维度: {len(emb)}, 向量前几维值: {emb[:5]}")
            except Exception as e:
                logger.warning(f"无法获取查询词向量: {str(e)}")
        
        # 执行原始查询
        docs = self.vectorstore.similarity_search_with_score(
            query, k=self.top_k, score_threshold=self.semantic_threshold
        )
        
        logger.info(f"查询 '{query}' 找到的文档数: {len(docs)}")
        if docs:
            logger.info(f"第一个文档的相似度分数: {docs[0][1]}")
            logger.info(f"第一个文档内容片段: {docs[0][0].page_content[:100]}...")
            
        return [doc for doc, _ in docs]
