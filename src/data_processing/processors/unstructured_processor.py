"""
非结构化文档处理模块，处理普通文本、PDF等非结构化文档。
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .base import BaseDocumentProcessor, ProcessorConfig
from ..vectorization.factory import VectorizationFactory
from ..splitting.splitter import DocumentSplitter

logger = logging.getLogger(__name__)

@dataclass
class UnstructuredConfig(ProcessorConfig):
    """非结构化文档处理器配置"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    enable_vectorization: bool = field(default_factory=lambda: os.getenv('ENABLE_VECTORIZATION', 'false').lower() == 'true')
    vectorization_method: str = field(default_factory=lambda: os.getenv('VECTORIZATION_METHOD', 'tfidf'))
    use_intelligent_chunking: bool = field(default_factory=lambda: os.getenv('USE_INTELLIGENT_CHUNKING', 'true').lower() == 'true')

class UnstructuredProcessor(BaseDocumentProcessor):
    """非结构化文档处理器实现，处理普通文本、PDF等非结构化文档。"""
    
    def __init__(self, config: Optional[UnstructuredConfig] = None):
        super().__init__(config or UnstructuredConfig())
        self.config = config or UnstructuredConfig()
        
        # 初始化文本分割器
        if self.config.use_intelligent_chunking:
            self.splitter = DocumentSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        else:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        
        # 初始化向量化器
        self.vectorizer = None
        if self.config.enable_vectorization:
            try:
                self.vectorizer = VectorizationFactory.create_vectorizer(
                    method=self.config.vectorization_method
                )
                logger.info(f"非结构化处理器初始化向量化器: {self.config.vectorization_method}")
            except Exception as e:
                logger.error(f"初始化向量化器失败: {str(e)}")
    
    def process_file(self, file_content: bytes, filename: str, mime_type: str) -> List[Document]:
        """处理文件。
        
        Args:
            file_content: 原始文件内容
            filename: 文件名
            mime_type: MIME类型
            
        Returns:
            处理后的文档列表
        """
        try:
            # 解码文件内容
            text_content = file_content.decode('utf-8', errors='ignore')
            
            # 处理文本内容
            return self.process_text(text_content, {"source": filename})
            
        except Exception as e:
            logger.error(f"处理文件时出错 {filename}: {str(e)}")
            return []
    
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """处理文本内容。
        
        Args:
            text: 要处理的文本内容
            metadata: 可选的元数据
            
        Returns:
            处理后的文档列表
        """
        try:
            # 创建基础文档
            base_doc = Document(
                page_content=text,
                metadata=metadata or {}
            )
            
            # 分割文档
            if self.config.use_intelligent_chunking:
                # 使用智能分块
                docs = self.splitter.split_text(text, metadata or {})
            else:
                # 使用传统分块
                docs = self.splitter.split_documents([base_doc])
            
            # 如果启用了向量化，为每个文档添加向量
            if self.vectorizer and self.config.enable_vectorization:
                for doc in docs:
                    doc.metadata["vector"] = self.vectorizer.vectorize(doc.page_content)
            
            return docs
            
        except Exception as e:
            logger.error(f"处理文本时出错: {str(e)}")
            return [] 