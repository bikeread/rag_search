"""
文档处理模块，支持多种文件格式。
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import mimetypes

from langchain.schema import Document
from .base import BaseDocumentProcessor, DocumentType, ProcessorConfig
from .unstructured_processor import UnstructuredProcessor, UnstructuredConfig
from .table_processor import TableProcessor, TableProcessorConfig
from ..vectorization.factory import VectorizationFactory

logger = logging.getLogger(__name__)

class DocumentProcessor(BaseDocumentProcessor):
    """文档处理器实现，支持多种格式。"""
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        super().__init__(config)
        
        # 是否启用向量化
        self.enable_vectorization = os.getenv('ENABLE_VECTORIZATION', 'false').lower() == 'true'
        logger.info(f"初始化DocumentProcessor，启用向量化: {self.enable_vectorization}")
        
        # 创建非结构化处理器，确保使用正确的配置类型
        unstructured_config = UnstructuredConfig()
        if config:
            # 从通用配置复制基础属性到非结构化配置
            unstructured_config.doc_type = config.doc_type
            unstructured_config.chunk_size = config.chunk_size
            unstructured_config.chunk_overlap = config.chunk_overlap
            unstructured_config.encoding = config.encoding
            unstructured_config.language = config.language
            unstructured_config.metadata = config.metadata
            unstructured_config.enable_vectorization = self.enable_vectorization
        
        self.unstructured_processor = UnstructuredProcessor(unstructured_config)
        
        # 创建表格处理器，并传入向量化配置
        table_config = TableProcessorConfig(
            max_rows_per_chunk=int(os.getenv('TABLE_MAX_ROWS_PER_CHUNK', '5')),
            preserve_empty_cells=os.getenv('TABLE_PRESERVE_EMPTY_CELLS', 'false').lower() == 'true',
            enable_vectorization=self.enable_vectorization
        )
        self.table_processor = TableProcessor(table_config)
    
    def process_file(self, file_content: bytes, filename: str, mime_type: str) -> List[Document]:
        """使用适当的处理器处理文件。
        
        Args:
            file_content: 原始文件内容
            filename: 文件名
            mime_type: MIME类型
            
        Returns:
            处理后的文档列表
        """
        try:
            # 检查是否是表格类型文件
            if self._is_table_content(file_content.decode('utf-8', errors='ignore')):
                logger.info(f"使用表格处理器处理文件: {filename}")
                return self.table_processor.process_file(file_content, filename, mime_type)
            
            # 使用通用处理器
            logger.info(f"使用通用处理器处理文件: {filename}")
            return self.unstructured_processor.process_file(file_content, filename, mime_type)
            
        except Exception as e:
            logger.error(f"处理文件时出错 {filename}: {str(e)}")
            return []
    
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """使用适当的处理器处理文本内容。
        
        Args:
            text: 要处理的文本内容
            metadata: 可选的元数据
            
        Returns:
            处理后的文档列表
        """
        try:
            # 检查是否是表格内容
            if self._is_table_content(text):
                logger.info("使用表格处理器处理文本")
                return self.table_processor.process_text(text, metadata)
            
            # 使用通用处理器
            logger.info("使用通用处理器处理文本")
            return self.unstructured_processor.process_text(text, metadata)
            
        except Exception as e:
            logger.error(f"处理文本时出错: {str(e)}")
            return []
    
    def _is_table_content(self, text: str) -> bool:
        """检查内容是否是表格格式
        
        Args:
            text: 文本内容
            
        Returns:
            是否是表格格式
        """
        try:
            lines = text.strip().split("\n")
            if len(lines) < 2:  # 至少需要表头和一行数据
                return False
            
            # 检查是否所有行都包含相同数量的分隔符
            pipe_counts = [line.count("|") for line in lines]
            if not all(count == pipe_counts[0] for count in pipe_counts):
                return False
            
            # 检查是否存在表格分隔行（---|---）
            for line in lines:
                if set(line.strip()) <= {"-", "+", "|"}:
                    return True
            
            # 检查每行的列数是否一致
            cells_per_row = [len(line.strip().strip("|").split("|")) for line in lines]
            return len(set(cells_per_row)) == 1 and cells_per_row[0] > 1
            
        except Exception as e:
            logger.warning(f"检查表格内容时出错: {str(e)}")
            return False

class DocumentProcessorFactory:
    """文档处理器工厂类。"""
    
    @staticmethod
    def create_processor(processor_type: str = "default", config: Optional[ProcessorConfig] = None) -> BaseDocumentProcessor:
        """创建文档处理器。
        
        Args:
            processor_type: 要创建的处理器类型
            config: 可选的处理器配置
            
        Returns:
            文档处理器实例
        """
        if processor_type == "table":
            return TableProcessor(config)
        else:
            return DocumentProcessor(config) 