"""
文档处理器核心逻辑
整合现有的document_processor和splitting逻辑，适配微服务架构
"""

import asyncio
import logging
import mimetypes
import uuid
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import tempfile
import os

from langchain.schema import Document

# 导入现有的处理器
from .processors.document_processor import DocumentProcessor, DocumentProcessorFactory
from .processors.base import ProcessorConfig, DocumentType
from .splitting.splitter import DocumentSplitter
# from .splitting.strategies import SplittingStrategy  # 使用DocumentSplitter已足够

logger = logging.getLogger(__name__)


class DocumentChunk:
    """文档分块数据类"""
    
    def __init__(self, 
                 chunk_id: str,
                 content: str,
                 chunk_index: int,
                 metadata: Dict[str, Any] = None):
        self.chunk_id = chunk_id
        self.content = content
        self.chunk_index = chunk_index
        self.metadata = metadata or {}
    
    def dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata
        }


class ProcessingResult:
    """处理结果数据类"""
    
    def __init__(self,
                 document_id: str,
                 filename: str,
                 chunks: List[DocumentChunk],
                 status: str = "completed",
                 error_message: str = None,
                 metadata: Dict[str, Any] = None):
        self.document_id = document_id
        self.filename = filename
        self.chunks = chunks
        self.status = status
        self.error_message = error_message
        self.metadata = metadata or {}
        
    def dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "chunks": [chunk.dict() for chunk in self.chunks],
            "chunk_count": len(self.chunks),
            "status": self.status,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


class ModernDocumentProcessor:
    """
    现代化文档处理器
    整合现有处理逻辑，提供微服务接口
    """
    
    def __init__(self):
        """初始化文档处理器"""
        logger.info("初始化现代化文档处理器")
        
        # 初始化配置
        self.config = ProcessorConfig()
        
        # 初始化文档处理器
        self.processor = DocumentProcessorFactory.create_processor("default", self.config)
        
        # 初始化文本分割器
        self.text_splitter = DocumentSplitter()
        
        # 支持的文件类型映射
        self.supported_types = {
            'application/pdf': DocumentType.PDF,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DocumentType.DOCX,
            'application/msword': DocumentType.DOCX,
            'text/plain': DocumentType.TXT,
            'text/markdown': DocumentType.TXT,
            'text/x-markdown': DocumentType.TXT,  # 另一种Markdown MIME类型
            'application/octet-stream': DocumentType.TXT,  # 通用二进制类型，通过文件扩展名判断
            'text/html': DocumentType.HTML,
            'application/vnd.ms-excel': DocumentType.EXCEL,
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': DocumentType.EXCEL,
            'text/csv': DocumentType.CSV
        }
        
        logger.info(f"支持的文件类型: {list(self.supported_types.keys())}")
    
    async def process_file(self,
                          file_content: bytes,
                          filename: str,
                          mime_type: str,
                          document_id: str,
                          enable_chunking: bool = True,
                          chunk_size: int = 1000,
                          chunk_overlap: int = 200) -> ProcessingResult:
        """
        处理文件的主入口
        
        Args:
            file_content: 文件内容
            filename: 文件名
            mime_type: MIME类型
            document_id: 文档ID
            enable_chunking: 是否启用分块
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
            
        Returns:
            ProcessingResult: 处理结果
        """
        logger.info(f"开始处理文件: {filename}, 类型: {mime_type}, 大小: {len(file_content)} bytes")
        
        try:
            # 1. 选择处理器
            document_type = self._get_document_type(mime_type, filename)
            logger.info(f"检测到文档类型: {document_type}")
            
            # 2. 提取文本内容
            documents = await self._extract_text_content(
                file_content, filename, mime_type, document_type
            )
            
            # 3. 分块处理（如果启用）
            chunks = []
            if enable_chunking and documents:
                chunks = await self._chunk_documents(
                    documents, document_id, chunk_size, chunk_overlap
                )
            else:
                # 不分块，直接转换为chunks
                for i, doc in enumerate(documents):
                    chunk = DocumentChunk(
                        chunk_id=f"{document_id}_chunk_{i}",
                        content=doc.page_content,
                        chunk_index=i,
                        metadata={
                            **doc.metadata,
                            "document_id": document_id,
                            "filename": filename,
                            "mime_type": mime_type
                        }
                    )
                    chunks.append(chunk)
            
            logger.info(f"处理完成: {filename}, 生成 {len(chunks)} 个分块")
            
            return ProcessingResult(
                document_id=document_id,
                filename=filename,
                chunks=chunks,
                status="completed",
                metadata={
                    "document_type": document_type.value,
                    "original_documents_count": len(documents),
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "enable_chunking": enable_chunking
                }
            )
            
        except Exception as e:
            logger.error(f"处理文件时出错: {str(e)}")
            return ProcessingResult(
                document_id=document_id,
                filename=filename,
                chunks=[],
                status="failed",
                error_message=str(e)
            )
    
    def _get_document_type(self, mime_type: str, filename: str) -> DocumentType:
        """
        根据MIME类型和文件名确定文档类型
        
        Args:
            mime_type: MIME类型
            filename: 文件名
            
        Returns:
            DocumentType: 文档类型
        """
        # 首先尝试从MIME类型匹配
        if mime_type in self.supported_types:
            return self.supported_types[mime_type]
        
        # 如果MIME类型不匹配，尝试从文件扩展名推断
        if filename:
            suffix = Path(filename).suffix.lower()
            type_mapping = {
                '.pdf': DocumentType.PDF,
                '.docx': DocumentType.DOCX,
                '.doc': DocumentType.DOCX,
                '.txt': DocumentType.TXT,
                '.md': DocumentType.TXT,
                '.html': DocumentType.HTML,
                '.htm': DocumentType.HTML,
                '.xlsx': DocumentType.EXCEL,
                '.xls': DocumentType.EXCEL,
                '.csv': DocumentType.CSV
            }
            
            if suffix in type_mapping:
                return type_mapping[suffix]
        
        # 默认作为文本处理
        logger.warning(f"未知文件类型 {mime_type}，默认按文本处理")
        return DocumentType.TXT
    
    async def _extract_text_content(self,
                                   file_content: bytes,
                                   filename: str,
                                   mime_type: str,
                                   document_type: DocumentType) -> List[Document]:
        """
        提取文本内容
        
        Args:
            file_content: 文件内容
            filename: 文件名
            mime_type: MIME类型
            document_type: 文档类型
            
        Returns:
            List[Document]: 文档列表
        """
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name
            
            try:
                # 使用现有的处理器处理文件
                documents = await asyncio.get_event_loop().run_in_executor(
                    None, self.processor.process_file, file_content, filename, mime_type
                )
                
                # 添加额外的元数据
                for doc in documents:
                    doc.metadata.update({
                        "filename": filename,
                        "mime_type": mime_type,
                        "document_type": document_type.value,
                        "file_size": len(file_content)
                    })
                
                return documents
                
            finally:
                # 清理临时文件
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"提取文本内容失败: {str(e)}")
            
            # 如果专用处理器失败，尝试作为文本处理
            if document_type != DocumentType.TXT:
                try:
                    text_content = file_content.decode('utf-8', errors='ignore')
                    return [Document(
                        page_content=text_content,
                        metadata={
                            "filename": filename,
                            "mime_type": mime_type,
                            "document_type": "text_fallback",
                            "file_size": len(file_content)
                        }
                    )]
                except Exception as fallback_error:
                    logger.error(f"文本回退处理也失败: {str(fallback_error)}")
            
            raise e
    
    async def _chunk_documents(self,
                             documents: List[Document],
                             document_id: str,
                             chunk_size: int,
                             chunk_overlap: int) -> List[DocumentChunk]:
        """
        分块处理文档
        
        Args:
            documents: 文档列表
            document_id: 文档ID
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
            
        Returns:
            List[DocumentChunk]: 分块列表
        """
        chunks = []
        global_chunk_index = 0
        
        try:
            for doc_index, document in enumerate(documents):
                # 使用文本分割器进行分块
                text_chunks = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._split_text,
                    document.page_content,
                    chunk_size,
                    chunk_overlap
                )
                
                # 转换为DocumentChunk对象
                for local_chunk_index, chunk_text in enumerate(text_chunks):
                    chunk = DocumentChunk(
                        chunk_id=f"{document_id}_chunk_{global_chunk_index}",
                        content=chunk_text,
                        chunk_index=global_chunk_index,
                        metadata={
                            **document.metadata,
                            "document_id": document_id,
                            "source_document_index": doc_index,
                            "local_chunk_index": local_chunk_index,
                            "chunk_size": chunk_size,
                            "chunk_overlap": chunk_overlap
                        }
                    )
                    chunks.append(chunk)
                    global_chunk_index += 1
            
            logger.info(f"分块完成: 从 {len(documents)} 个文档生成 {len(chunks)} 个分块")
            return chunks
            
        except Exception as e:
            logger.error(f"分块处理失败: {str(e)}")
            raise
    
    def _split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        分割文本
        
        Args:
            text: 要分割的文本
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
            
        Returns:
            List[str]: 分块文本列表
        """
        try:
            # 创建合适配置的分割器实例
            splitter = DocumentSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # 执行分割
            chunks = splitter.split_text(text)
            return chunks
            
        except Exception as e:
            logger.error(f"文本分割失败: {str(e)}")
            # 简单回退：按字符数分割
            chunks = []
            for i in range(0, len(text), chunk_size - chunk_overlap):
                chunk = text[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
            return chunks
    
    def get_supported_types(self) -> List[str]:
        """获取支持的文件类型列表"""
        return list(self.supported_types.keys())
    
    def is_supported_type(self, mime_type: str) -> bool:
        """检查是否支持指定的文件类型"""
        return mime_type in self.supported_types
