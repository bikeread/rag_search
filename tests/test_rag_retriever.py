import unittest
from unittest.mock import Mock, patch
import logging
from datetime import datetime
from langchain_core.documents import Document
from src.rag.retriever import RAGRetriever, VectorStoreRetriever
from src.data_processing.storage.base import VectorStoreBase

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRAGRetriever(unittest.TestCase):
    """测试RAG检索器的功能"""

    def setUp(self):
        """测试前的准备工作"""
        # 创建模拟的向量存储
        self.mock_vectorstore = Mock(spec=VectorStoreBase)
        
        # 设置模拟文档
        self.test_docs = [
            Document(
                page_content="ThinkPad X1 Carbon 2024款 | 价格: 12999元 | 处理器: Intel i7-1360P",
                metadata={"score": 0.95, "id": "doc1"}
            ),
            Document(
                page_content="MacBook Pro 14寸 M3 Pro | 价格: 18999元 | 处理器: M3 Pro",
                metadata={"score": 0.85, "id": "doc2"}
            ),
            Document(
                page_content="ROG 魔霸新锐 2024 | 价格: 9999元 | 处理器: AMD R9-7945HX",
                metadata={"score": 0.75, "id": "doc3"}
            )
        ]
        
        # 配置向量存储的行为
        self.mock_vectorstore.search.return_value = self.test_docs
        self.mock_vectorstore.similarity_search_with_score.return_value = [
            (doc, doc.metadata["score"]) for doc in self.test_docs
        ]
        
        # 创建检索器
        self.vector_retriever = VectorStoreRetriever(
            vectorstore=self.mock_vectorstore,
            semantic_threshold=0.7,
            top_k=3
        )
        
        # 创建RAG检索器
        self.rag_retriever = RAGRetriever(
            retriever=self.vector_retriever,
            config={
                "query_decomposition_enabled": True,
                "query_expansion_enabled": True
            }
        )

    def test_generate_answer_basic(self):
        """测试基本的问答功能"""
        # 准备测试数据
        query = "最贵的笔记本电脑多少钱"
        
        # 执行测试
        result = self.rag_retriever.generate_answer(
            query=query,
            chain_type="qa",
            temperature=0.7
        )
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)
        self.assertIn("sources", result)
        
        # 打印结果以供分析
        logger.info(f"Query: {query}")
        logger.info(f"Answer: {result['answer']}")
        logger.info(f"Sources: {result['sources']}")

    def test_generate_answer_with_chat_history(self):
        """测试带有对话历史的问答功能"""
        # 准备测试数据
        query = "它的处理器是什么"
        chat_history = [
            ("最贵的笔记本电脑是哪个", "根据数据，最贵的笔记本电脑是MacBook Pro 14寸 M3 Pro，售价18999元。")
        ]
        
        # 执行测试
        result = self.rag_retriever.generate_answer(
            query=query,
            chat_history=chat_history,
            chain_type="conversational",
            temperature=0.7
        )
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)
        self.assertIn("sources", result)
        
        # 打印结果以供分析
        logger.info(f"Query: {query}")
        logger.info(f"Chat History: {chat_history}")
        logger.info(f"Answer: {result['answer']}")
        logger.info(f"Sources: {result['sources']}")

    def test_generate_answer_no_results(self):
        """测试没有找到相关文档的情况"""
        # 修改向量存储的行为，返回空结果
        self.mock_vectorstore.search.return_value = []
        self.mock_vectorstore.similarity_search_with_score.return_value = []
        
        # 准备测试数据
        query = "显示器多少钱"
        
        # 执行测试
        result = self.rag_retriever.generate_answer(
            query=query,
            chain_type="qa",
            temperature=0.7
        )
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)
        self.assertEqual(len(result["sources"]), 0)
        self.assertIn("没有找到", result["answer"].lower())

    def test_generate_answer_error_handling(self):
        """测试错误处理功能"""
        # 修改向量存储的行为，抛出异常
        self.mock_vectorstore.search.side_effect = Exception("模拟的搜索错误")
        
        # 准备测试数据
        query = "笔记本电脑价格"
        
        # 执行测试
        result = self.rag_retriever.generate_answer(
            query=query,
            chain_type="qa",
            temperature=0.7
        )
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)
        self.assertIn("error_detail", result)
        self.assertIn("技术问题", result["answer"])

    def test_generate_answer_with_invalid_temperature(self):
        """测试无效的temperature参数处理"""
        # 准备测试数据
        query = "笔记本电脑推荐"
        
        # 执行测试 - temperature超出范围
        result = self.rag_retriever.generate_answer(
            query=query,
            chain_type="qa",
            temperature=1.5  # 无效的temperature值
        )
        
        # 验证结果
        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)
        self.assertIn("sources", result)

if __name__ == '__main__':
    unittest.main() 