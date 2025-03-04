"""
文档分割器实现，支持不同类型文档的智能分块。
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from langchain.schema import Document
from dataclasses import dataclass

from ..vectorization.base import BaseVectorizer

logger = logging.getLogger(__name__)

class DocumentType(str, Enum):
    """文档类型枚举。"""
    
    AUTO = "auto"
    SHORT_TEXT = "short_text"
    NORMAL_TEXT = "normal_text"
    LONG_TEXT = "long_text"
    CODE = "code"
    LEGAL = "legal"
    MEDICAL = "medical"
    CHINESE = "chinese"
    ENGLISH = "english"

@dataclass
class ChunkingPreset:
    """分块预设参数。"""
    
    chunk_size: int
    chunk_overlap: int
    separators: List[str]

# 不同文档类型的预设参数
CHUNKING_PRESETS = {
    DocumentType.AUTO: ChunkingPreset(1024, 200, ["\n\n", "\n", ". ", "。", " ", ""]),
    DocumentType.SHORT_TEXT: ChunkingPreset(384, 40, ["\n", ". ", "。", " ", ""]),
    DocumentType.NORMAL_TEXT: ChunkingPreset(768, 120, ["\n\n", "\n", ". ", "。", " ", ""]),
    DocumentType.LONG_TEXT: ChunkingPreset(1536, 300, ["\n\n", "\n", ". ", "。", " ", ""]),
    DocumentType.CODE: ChunkingPreset(512, 50, ["\n\n", "\n", ";", "{", "}", ""]),
    DocumentType.LEGAL: ChunkingPreset(2048, 500, ["\n\n", "\n", ". ", "。", " ", ""]),
    DocumentType.MEDICAL: ChunkingPreset(1024, 200, ["\n\n", "\n", ". ", "。", " ", ""]),
    DocumentType.CHINESE: ChunkingPreset(1024, 200, ["\n\n", "\n", "。", "！", "？", " ", ""]),
    DocumentType.ENGLISH: ChunkingPreset(512, 80, ["\n\n", "\n", ". ", "! ", "? ", " ", ""]),
}

class DocumentSplitter:
    """文档分割器，支持不同类型文档的分块。"""
    
    def __init__(
        self,
        doc_type: Union[str, DocumentType] = DocumentType.AUTO,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
        vectorizer: Optional[BaseVectorizer] = None
    ):
        """初始化文档分割器。
        
        Args:
            doc_type: 文档类型
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小
            separators: 分隔符列表
            vectorizer: 向量化器实例
        """
        # 确保doc_type是DocumentType类型
        if isinstance(doc_type, str):
            doc_type = DocumentType(doc_type.lower())
            
        # 获取预设参数
        preset = CHUNKING_PRESETS.get(doc_type, CHUNKING_PRESETS[DocumentType.AUTO])
        
        # 使用自定义参数覆盖预设参数
        self.chunk_size = chunk_size or preset.chunk_size
        self.chunk_overlap = chunk_overlap or preset.chunk_overlap
        self.separators = separators or preset.separators
        
        # 向量化器
        self.vectorizer = vectorizer
        
        logger.info(f"初始化文档分割器，类型: {doc_type}, 分块大小: {self.chunk_size}, 重叠大小: {self.chunk_overlap}")
    
    def split_text(self, text: str) -> List[str]:
        """分割文本为块。
        
        Args:
            text: 文本内容
            
        Returns:
            分割后的文本块列表
        """
        if not text:
            return []
            
        # 如果文本长度小于分块大小，直接返回整个文本
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # 计算结束位置
            end = start + self.chunk_size
            
            # 如果结束位置超出文本长度，直接到结尾
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # 尝试找到合适的分隔符
            found_separator = False
            
            for separator in self.separators:
                if not separator:
                    continue
                    
                # 找最接近结束位置的分隔符
                separator_position = text.rfind(separator, start, end)
                
                if separator_position != -1:
                    # 分隔符的位置加上分隔符的长度作为实际结束位置
                    actual_end = separator_position + len(separator)
                    chunks.append(text[start:actual_end])
                    start = actual_end - self.chunk_overlap
                    found_separator = True
                    break
            
            # 如果没有找到合适的分隔符，就强制分割
            if not found_separator:
                chunks.append(text[start:end])
                start = end - self.chunk_overlap
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表。
        
        Args:
            documents: 文档列表
            
        Returns:
            分割后的文档列表
        """
        result = []
        
        for doc in documents:
            # 分割文本内容
            text_chunks = self.split_text(doc.page_content)
            
            # 为每个文本块创建新的文档
            for i, chunk in enumerate(text_chunks):
                # 复制元数据
                metadata = dict(doc.metadata)
                metadata["chunk"] = i
                metadata["total_chunks"] = len(text_chunks)
                
                # 创建新文档
                new_doc = Document(
                    page_content=chunk,
                    metadata=metadata
                )
                
                # 如果有向量化器，添加向量
                if self.vectorizer:
                    vector = self.vectorizer.vectorize(chunk)
                    metadata["vector"] = vector
                
                result.append(new_doc)
        
        logger.info(f"文档分割: {len(documents)}个文档分割为{len(result)}个块")
        return result 