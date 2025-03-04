"""
基础文档处理器实现。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
import os
from langchain.schema import Document

class DocumentType(str, Enum):
    """文档类型枚举。"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"
    IMAGE = "image"
    EXCEL = "excel"
    CSV = "csv"
    AUTO = "auto"

@dataclass
class ProcessorConfig:
    """文档处理器基础配置。"""
    
    # 文档类型
    doc_type: DocumentType = DocumentType.AUTO
    
    # 分块设置
    chunk_size: int = field(default_factory=lambda: int(os.getenv('CHUNK_SIZE', '1024')))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv('CHUNK_OVERLAP', '200')))
    
    # 编码和语言
    encoding: str = field(default_factory=lambda: os.getenv('DEFAULT_ENCODING', 'utf-8'))
    language: str = field(default_factory=lambda: os.getenv('DEFAULT_LANGUAGE', 'zh'))
    
    # 附加元数据
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseDocumentProcessor(ABC):
    """文档处理器基类。"""
    
    def __init__(self, config: Optional[ProcessorConfig] = None):
        """使用配置初始化处理器。
        
        Args:
            config: 处理器配置
        """
        self.config = config or ProcessorConfig()
        self._setup_logging()
    
    def _setup_logging(self):
        """设置处理器日志。"""
        import logging
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def process_file(self, file_content: bytes, filename: str, mime_type: str) -> List[Document]:
        """处理单个文件。
        
        Args:
            file_content: 原始文件内容
            filename: 文件名
            mime_type: MIME类型
            
        Returns:
            处理后的文档列表
        """
        pass
    
    @abstractmethod
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """处理原始文本。
        
        Args:
            text: 要处理的文本内容
            metadata: 可选的元数据
            
        Returns:
            处理后的文档列表
        """
        pass
    
    def _add_metadata(self, doc: Document, additional_metadata: Optional[Dict[str, Any]] = None) -> Document:
        """向文档添加元数据。
        
        Args:
            doc: 要添加元数据的文档
            additional_metadata: 要添加的额外元数据
            
        Returns:
            添加了元数据的文档
        """
        # 从处理器配置开始
        metadata = dict(self.config.metadata)
        
        # 添加文档类型
        metadata["doc_type"] = self.config.doc_type.value
        
        # 添加额外元数据
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # 更新文档元数据
        if doc.metadata:
            doc.metadata.update(metadata)
        else:
            doc.metadata = metadata
            
        return doc 