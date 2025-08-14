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
                             chunk_overlap: int,
                             enable_semantic_chunking: bool = True) -> List[DocumentChunk]:
        """
        分块处理文档 - 增强版，支持语义感知分块和质量评估
        
        Args:
            documents: 文档列表
            document_id: 文档ID
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
            enable_semantic_chunking: 是否启用语义分块
            
        Returns:
            List[DocumentChunk]: 分块列表
        """
        chunks = []
        global_chunk_index = 0
        
        try:
            # 导入语义分块器
            from .splitting.semantic_chunker import EmbeddingBasedSemanticChunker
            from .splitting.chunk_quality_assessor import ChunkQualityAssessor, ChunkQualityReporter
            
            # 初始化语义分块器（如果启用）
            semantic_chunker = None
            if enable_semantic_chunking:
                try:
                    semantic_chunker = EmbeddingBasedSemanticChunker(
                        vector_service_url="http://localhost:8002",
                        min_chunk_size=max(200, chunk_size // 2),
                        max_chunk_size=min(2000, chunk_size * 2)
                    )
                    logger.info("语义分块器初始化成功")
                except Exception as e:
                    logger.warning(f"语义分块器初始化失败，回退到传统分块: {str(e)}")
                    semantic_chunker = None
            
            # 初始化质量评估器
            quality_assessor = ChunkQualityAssessor()
            
            for doc_index, document in enumerate(documents):
                doc_chunks = []
                
                if semantic_chunker:
                    # 使用语义感知分块
                    try:
                        semantic_chunks = await semantic_chunker.chunk_document(
                            text=document.page_content,
                            metadata=document.metadata
                        )
                        
                        # 转换为DocumentChunk对象
                        for local_chunk_index, semantic_chunk in enumerate(semantic_chunks):
                            chunk = DocumentChunk(
                                chunk_id=f"{document_id}_chunk_{global_chunk_index}",
                                content=semantic_chunk.text,
                                chunk_index=global_chunk_index,
                                metadata={
                                    **document.metadata,
                                    **semantic_chunk.metadata,
                                    "document_id": document_id,
                                    "source_document_index": doc_index,
                                    "local_chunk_index": local_chunk_index,
                                    "semantic_coherence": semantic_chunk.semantic_coherence_score,
                                    "contains_critical_info": semantic_chunk.contains_critical_info,
                                    "chunking_method": "semantic_embedding"
                                }
                            )
                            doc_chunks.append(chunk)
                            global_chunk_index += 1
                        
                        logger.info(f"文档 {doc_index} 语义分块完成: {len(semantic_chunks)} 个分块")
                        
                    except Exception as e:
                        logger.error(f"语义分块失败，回退到智能分块: {str(e)}")
                        # 回退到智能分块
                        doc_chunks = await self._fallback_smart_chunking(
                            document, document_id, doc_index, chunk_size, chunk_overlap, global_chunk_index
                        )
                        global_chunk_index += len(doc_chunks)
                else:
                    # 使用传统智能分块
                    doc_chunks = await self._fallback_smart_chunking(
                        document, document_id, doc_index, chunk_size, chunk_overlap, global_chunk_index
                    )
                    global_chunk_index += len(doc_chunks)
                
                chunks.extend(doc_chunks)
            
            # 评估分块质量
            await self._evaluate_and_log_quality(documents, chunks, quality_assessor)
            
            logger.info(f"分块完成: 从 {len(documents)} 个文档生成 {len(chunks)} 个分块")
            return chunks
            
        except Exception as e:
            logger.error(f"分块处理失败: {str(e)}")
            raise
    
    async def _fallback_smart_chunking(self,
                                     document: Document,
                                     document_id: str,
                                     doc_index: int,
                                     chunk_size: int,
                                     chunk_overlap: int,
                                     start_chunk_index: int) -> List[DocumentChunk]:
        """降级的智能分块策略"""
        chunks = []
        
        # 优化分块参数
        optimized_chunk_size, optimized_overlap = self._optimize_chunk_parameters(
            document.page_content, chunk_size, chunk_overlap
        )
        
        # 使用智能分块策略
        smart_chunks = await asyncio.get_event_loop().run_in_executor(
            None,
            self._smart_split_text,
            document.page_content,
            optimized_chunk_size,
            optimized_overlap
        )
        
        # 转换为DocumentChunk对象
        for local_chunk_index, chunk_data in enumerate(smart_chunks):
            chunk_text = chunk_data['text'] if isinstance(chunk_data, dict) else chunk_data
            chunk_metadata = chunk_data.get('metadata', {}) if isinstance(chunk_data, dict) else {}
            
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_chunk_{start_chunk_index + local_chunk_index}",
                content=chunk_text,
                chunk_index=start_chunk_index + local_chunk_index,
                metadata={
                    **document.metadata,
                    **chunk_metadata,
                    "document_id": document_id,
                    "source_document_index": doc_index,
                    "local_chunk_index": local_chunk_index,
                    "chunk_size": optimized_chunk_size,
                    "chunk_overlap": optimized_overlap,
                    "chunking_method": "smart_fallback"
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _evaluate_and_log_quality(self,
                                      documents: List[Document],
                                      chunks: List[DocumentChunk],
                                      quality_assessor) -> None:
        """评估并记录分块质量"""
        try:
            # 准备评估数据
            original_text = "\n\n".join([doc.page_content for doc in documents])
            chunk_data = [
                {
                    "text": chunk.content,
                    "metadata": chunk.metadata
                }
                for chunk in chunks
            ]
            
            # 执行质量评估
            quality_metrics = await quality_assessor.assess_chunking_quality(
                original_text=original_text,
                chunks=chunk_data,
                chunk_method=chunks[0].metadata.get("chunking_method", "unknown") if chunks else "unknown"
            )
            
            # 生成质量报告
            from .splitting.chunk_quality_assessor import ChunkQualityReporter
            reporter = ChunkQualityReporter()
            quality_report = reporter.generate_quality_report(quality_metrics)
            
            # 记录质量评估结果
            logger.info(f"分块质量评估完成:")
            logger.info(f"  总体评分: {quality_report['overall_assessment']['score']}")
            logger.info(f"  质量等级: {quality_report['overall_assessment']['level']}")
            logger.info(f"  语义连贯性: {quality_report['dimension_scores']['semantic_coherence']}")
            logger.info(f"  信息保留度: {quality_report['dimension_scores']['information_preservation']}")
            logger.info(f"  边界质量: {quality_report['dimension_scores']['boundary_quality']}")
            
            # 如果质量较差，记录改进建议
            if quality_metrics.overall_score < 0.7:
                logger.warning("分块质量较差，建议优化:")
                for recommendation in quality_report['recommendations']:
                    logger.warning(f"  - {recommendation}")
            
        except Exception as e:
            logger.error(f"质量评估失败: {str(e)}")
            # 不影响主流程，继续执行
    
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
    
    def _optimize_chunk_parameters(self, text: str, default_chunk_size: int, default_overlap: int) -> tuple:
        """
        优化分块参数，根据文本特征调整分块策略
        
        Args:
            text: 文本内容
            default_chunk_size: 默认分块大小
            default_overlap: 默认重叠大小
            
        Returns:
            tuple: (优化后的chunk_size, 优化后的overlap)
        """
        import re
        
        # 分析文本特征
        text_length = len(text)
        
        # 数值密度分析
        number_pattern = r'\d+\.?\d*[%万亿千百十]?|[0-9]+[\.,][0-9]+|\d+:\d+|\d{4}[-/]\d{1,2}[-/]\d{1,2}'
        numbers = re.findall(number_pattern, text)
        number_density = len(numbers) / max(text_length, 1) * 1000  # 每1000字符的数字数量
        
        # GitHub项目信息分析（特殊优化）
        has_github_info = bool(re.search(r'(star|fork|⭐|GitHub|项目|repository)', text, re.IGNORECASE))
        has_project_comparison = bool(re.search(r'(比较|对比|数量|排行|最多|最少)', text))
        
        # 表格和列表结构分析
        has_structured_data = bool(re.search(r'(\n\s*\|\s*|\n\s*[-*+]\s*|\n\s*\d+\.?\s*)', text))
        
        optimized_chunk_size = default_chunk_size
        optimized_overlap = default_overlap
        
        # 针对数值密集内容的优化
        if number_density > 5:  # 高数值密度
            optimized_chunk_size = int(default_chunk_size * 0.7)  # 减小30%
            optimized_overlap = int(default_overlap * 1.5)  # 增加50%重叠
            logger.info(f"检测到高数值密度({number_density:.1f}/1000字符)，调整分块参数")
        
        # 针对GitHub项目信息的特殊优化
        if has_github_info and has_project_comparison:
            optimized_chunk_size = int(default_chunk_size * 0.6)  # 更小的块保证数字关联性
            optimized_overlap = int(default_overlap * 2.0)  # 更大重叠保证比较信息完整
            logger.info("检测到GitHub项目比较信息，应用专门优化")
        
        # 针对结构化数据的优化
        if has_structured_data:
            optimized_chunk_size = int(default_chunk_size * 0.8)  # 适度减小
            optimized_overlap = int(default_overlap * 1.3)  # 适度增加重叠
            logger.info("检测到结构化数据，调整分块策略")
        
        # 确保参数合理性
        optimized_chunk_size = max(200, min(optimized_chunk_size, default_chunk_size * 2))
        optimized_overlap = max(50, min(optimized_overlap, optimized_chunk_size // 3))
        
        return optimized_chunk_size, optimized_overlap
    
    def _smart_split_text(self, text: str, chunk_size: int, chunk_overlap: int):
        """
        智能文本分割，支持语义感知分块
        
        Args:
            text: 文本内容
            chunk_size: 分块大小
            chunk_overlap: 重叠大小
            
        Returns:
            分块结果列表
        """
        try:
            # 尝试使用语义感知分块
            from .splitting.strategies import SplitStrategy
            
            # 先检查是否值得使用高级分块
            if len(text) > chunk_size and self._has_critical_content(text):
                semantic_chunks = SplitStrategy.semantic_aware_split(
                    text=text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    preserve_numbers=True
                )
                
                if semantic_chunks:
                    logger.info(f"使用语义感知分块，生成{len(semantic_chunks)}个块")
                    return semantic_chunks
            
            # 回退到改进的分隔符策略
            separators = self._get_optimal_separators(text)
            chunks = self._split_with_separators(text, chunk_size, chunk_overlap, separators)
            
            return [{'text': chunk, 'metadata': {'split_method': 'separator_optimized'}} for chunk in chunks]
            
        except Exception as e:
            logger.warning(f"智能分块失败，回退到基本分块: {str(e)}")
            # 最后的回退策略
            return self._split_text(text, chunk_size, chunk_overlap)
    
    def _has_critical_content(self, text: str) -> bool:
        """检查文本是否包含关键内容需要特殊处理"""
        import re
        
        # 数字和数值信息
        number_patterns = [
            r'\d+\.?\d*[%万亿千百十]',    # 带单位的数字
            r'\d+\s*[⭐★star]',           # star数量
            r'\d+\s*[fork]',             # fork数量
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # 日期
            r'\$\d+\.?\d*',              # 货币
        ]
        
        has_numbers = any(re.search(pattern, text, re.IGNORECASE) for pattern in number_patterns)
        
        # 比较和排序信息
        comparison_keywords = ['最多', '最少', '比较', '对比', '排行', '数量', '大小', '高低']
        has_comparison = any(keyword in text for keyword in comparison_keywords)
        
        return has_numbers or has_comparison
    
    def _get_optimal_separators(self, text: str) -> List[str]:
        """根据文本特征获取最优分隔符"""
        import re
        
        # 基础分隔符
        separators = ["\n\n", "\n"]
        
        # 检测中文内容
        chinese_ratio = len(re.findall(r'[\u4e00-\u9fff]', text)) / max(len(text), 1)
        if chinese_ratio > 0.5:
            separators.extend(["。", "！", "？", "；"])
        else:
            separators.extend([". ", "! ", "? ", "; "])
        
        # 检测技术文档特征
        if re.search(r'(API|HTTP|JSON|XML|function|class|def)', text, re.IGNORECASE):
            separators.extend([";", "{", "}", "class ", "def ", "function "])
        
        # 检测列表和项目信息
        if re.search(r'(\n\s*[-*+•]\s*|\n\s*\d+\.?\s*)', text):
            separators.extend(["\n- ", "\n* ", "\n+ ", "\n• "])
        
        separators.append(" ")  # 最后的分隔符
        return separators
    
    def _split_with_separators(self, text: str, chunk_size: int, chunk_overlap: int, separators: List[str]) -> List[str]:
        """使用分隔符分割文本的优化版本"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                chunk = text[start:]
                if chunk.strip():
                    chunks.append(chunk.strip())
                break
            
            # 寻找最佳分割点
            best_split = end
            for separator in separators:
                if not separator:
                    continue
                
                # 在合理范围内寻找分隔符
                search_start = max(start, end - chunk_size // 4)
                separator_pos = text.rfind(separator, search_start, end)
                
                if separator_pos != -1:
                    best_split = separator_pos + len(separator)
                    break
            
            chunk = text[start:best_split].strip()
            if chunk:
                chunks.append(chunk)
            
            # 计算下一个起始位置，考虑重叠
            start = max(best_split - chunk_overlap, start + 1)
        
        return chunks
    
    def _assess_chunking_quality(self, documents: List[Document], chunks: List[DocumentChunk]) -> float:
        """
        评估分块质量
        
        Args:
            documents: 原始文档
            chunks: 分块结果
            
        Returns:
            float: 质量分数 (0-1)
        """
        if not chunks:
            return 0.0
        
        import re
        
        # 计算各项指标
        total_original_length = sum(len(doc.page_content) for doc in documents)
        total_chunk_length = sum(len(chunk.content) for chunk in chunks)
        
        # 1. 长度保持率 (期望接近1)
        length_retention = min(total_chunk_length / max(total_original_length, 1), 1.0)
        
        # 2. 数字信息保护率
        original_numbers = set()
        chunk_numbers = set()
        
        number_pattern = r'\d+\.?\d*[%万亿千百十⭐]?|\d+:\d+|\d{4}[-/]\d{1,2}[-/]\d{1,2}'
        
        for doc in documents:
            original_numbers.update(re.findall(number_pattern, doc.page_content))
        
        for chunk in chunks:
            chunk_numbers.update(re.findall(number_pattern, chunk.content))
        
        number_preservation = (
            len(original_numbers & chunk_numbers) / max(len(original_numbers), 1) 
            if original_numbers else 1.0
        )
        
        # 3. 块大小一致性
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        if chunk_sizes:
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            size_variance = sum((size - avg_size) ** 2 for size in chunk_sizes) / len(chunk_sizes)
            size_consistency = max(0, 1 - (size_variance ** 0.5) / avg_size)
        else:
            size_consistency = 0
        
        # 4. 语义完整性（简化评估）
        complete_chunks = sum(
            1 for chunk in chunks 
            if chunk.content.strip() and 
               (chunk.content.strip()[-1] in '.!?。！？' or len(chunk.content) < 100)
        )
        semantic_completeness = complete_chunks / len(chunks)
        
        # 加权计算总分
        weights = {
            'length_retention': 0.2,
            'number_preservation': 0.4,  # 数字保护最重要
            'size_consistency': 0.2,
            'semantic_completeness': 0.2
        }
        
        quality_score = (
            length_retention * weights['length_retention'] +
            number_preservation * weights['number_preservation'] +
            size_consistency * weights['size_consistency'] +
            semantic_completeness * weights['semantic_completeness']
        )
        
        logger.info(f"分块质量评估 - 长度保持: {length_retention:.2f}, "
                   f"数字保护: {number_preservation:.2f}, "
                   f"大小一致性: {size_consistency:.2f}, "
                   f"语义完整性: {semantic_completeness:.2f}")
        
        return quality_score

    def get_supported_types(self) -> List[str]:
        """获取支持的文件类型列表"""
        return list(self.supported_types.keys())
    
    def is_supported_type(self, mime_type: str) -> bool:
        """检查是否支持指定的文件类型"""
        return mime_type in self.supported_types
