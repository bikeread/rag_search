"""
Tests for the RAG-based Q&A system.
"""

import unittest
from src.rag.retriever import RAGRetriever
from src.data_processing.splitting.splitter import DocumentSplitter
from src.data_processing.storage.base import VectorStoreConfig, VectorStoreType
from src.data_processing.storage.factory import VectorStoreFactory
from config.config import (
    VECTOR_DB_TYPE,
    VECTOR_DB_PATH,
    EMBEDDING_MODEL,
    MILVUS_HOST,
    MILVUS_PORT,
    MILVUS_COLLECTION
)

class TestRAGSystem(unittest.TestCase):
    """Test cases for the RAG system."""
    
    def setUp(self):
        """Set up test environment."""
        self.retriever = RAGRetriever()
        self.splitter = DocumentSplitter()
        
        # 创建向量存储配置
        connection_args = {
            "vector_db_path": VECTOR_DB_PATH.replace(".milvus", ".test.milvus")
        }
        
        # 如果是 Milvus，添加 Milvus 专用配置
        if VECTOR_DB_TYPE == "milvus":
            connection_args.update({
                "host": MILVUS_HOST,
                "port": MILVUS_PORT,
                "collection_name": f"test_{MILVUS_COLLECTION}"  # 使用测试专用的集合名
            })
        
        vector_store_config = VectorStoreConfig(
            store_type=VectorStoreType(VECTOR_DB_TYPE),
            model_name=EMBEDDING_MODEL,
            connection_args=connection_args
        )
        
        # 使用工厂创建向量存储
        self.vector_store = VectorStoreFactory.create_store(vector_store_config)
        
        # Sample test documents
        self.test_docs = [
            "Python is a high-level programming language.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing deals with text and speech."
        ]
    
    def test_document_splitting(self):
        """Test document splitting functionality."""
        chunks = self.splitter.split_documents(self.test_docs)
        self.assertTrue(len(chunks) >= len(self.test_docs))
    
    def test_vector_store(self):
        """Test vector store operations."""
        # Add documents
        self.vector_store.add_documents(self.test_docs)
        
        # Test search
        results = self.vector_store.search("What is Python?")
        self.assertTrue(len(results) > 0)
        self.assertIn("Python", results[0])
    
    def test_rag_generation(self):
        """Test RAG answer generation."""
        # Add knowledge base
        self.retriever.add_documents(self.test_docs)
        
        # Test question answering
        question = "What is Python?"
        answer = self.retriever.generate_answer(question)
        self.assertTrue(isinstance(answer, dict))
        self.assertIn("answer", answer)
        self.assertIn("Python", answer["answer"])

if __name__ == '__main__':
    unittest.main() 