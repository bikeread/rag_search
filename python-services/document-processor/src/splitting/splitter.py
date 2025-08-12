"""
文档分割器实现，支持不同类型文档的智能分块。
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from langchain.schema import Document
from dataclasses import dataclass

# from ..vectorization.base import BaseVectorizer  # 在微服务架构中，向量化由vector-service提供

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

# 不同文档类型的预设参数 - 优化后的配置
CHUNKING_PRESETS = {
    DocumentType.AUTO: ChunkingPreset(1024, 200, ["\n\n", "\n", ". ", "。", " ", ""]),
    DocumentType.SHORT_TEXT: ChunkingPreset(512, 80, ["\n", ". ", "。", " ", ""]),
    DocumentType.NORMAL_TEXT: ChunkingPreset(1000, 200, ["\n\n", "\n", ". ", "。", " ", ""]),
    DocumentType.LONG_TEXT: ChunkingPreset(1536, 300, ["\n\n", "\n", ". ", "。", " ", ""]),
    DocumentType.CODE: ChunkingPreset(800, 100, ["\n\n", "\n", ";", "{", "}", ""]),
    DocumentType.LEGAL: ChunkingPreset(1200, 250, ["\n\n", "\n", ". ", "。", " ", ""]),  # 减小以保护数字
    DocumentType.MEDICAL: ChunkingPreset(1000, 200, ["\n\n", "\n", ". ", "。", " ", ""]),
    DocumentType.CHINESE: ChunkingPreset(1000, 200, ["\n\n", "\n", "。", "！", "？", " ", ""]),
    DocumentType.ENGLISH: ChunkingPreset(800, 160, ["\n\n", "\n", ". ", "! ", "? ", " ", ""]),  # 增加重叠
}

class DocumentSplitter:
    """文档分割器，支持不同类型文档的分块。"""
    
    def __init__(
        self,
        doc_type: Union[str, DocumentType] = DocumentType.AUTO,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
        vectorizer: Optional[Any] = None  # BaseVectorizer在微服务架构中由vector-service提供
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
    
    def split_text_semantic(self, text: str, enable_semantic_protection: bool = True) -> List[Dict[str, Any]]:
        """使用语义感知策略分割文本，保护重要信息。
        
        Args:
            text: 文本内容
            enable_semantic_protection: 是否启用语义保护
            
        Returns:
            分割后的文本块列表，包含元数据
        """
        if not text:
            return []
            
        # 导入分割策略
        from .strategies import SplitStrategy
        
        if enable_semantic_protection:
            # 使用语义感知分块
            return SplitStrategy.semantic_aware_split(
                text=text,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                preserve_numbers=True
            )
        else:
            # 使用自适应分块
            return SplitStrategy.adaptive_split(
                text=text,
                target_chunk_size=self.chunk_size,
                overlap_ratio=self.chunk_overlap / self.chunk_size if self.chunk_size > 0 else 0.2
            )
    
    def split_text_with_quality_assessment(self, text: str) -> Dict[str, Any]:
        """分割文本并评估分块质量。
        
        Args:
            text: 文本内容
            
        Returns:
            分割结果和质量评估
        """
        if not text:
            return {'chunks': [], 'quality': {'score': 0, 'metrics': {}}}
            
        # 执行语义感知分块
        chunks = self.split_text_semantic(text, enable_semantic_protection=True)
        
        # 评估分块质量
        quality_metrics = self._assess_chunking_quality(text, chunks)
        
        return {
            'chunks': chunks,
            'quality': quality_metrics,
            'original_length': len(text),
            'chunk_count': len(chunks)
        }
    
    def _assess_chunking_quality(self, original_text: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估分块质量。
        
        Args:
            original_text: 原始文本
            chunks: 分块结果
            
        Returns:
            质量评估结果
        """
        if not chunks:
            return {'score': 0, 'metrics': {'error': 'No chunks generated'}}
        
        metrics = {}
        
        # 1. 数字信息保护率
        original_numbers = self._extract_numbers(original_text)
        chunks_numbers = []
        for chunk in chunks:
            chunk_numbers = self._extract_numbers(chunk['text'])
            chunks_numbers.extend(chunk_numbers)
        
        if original_numbers:
            number_preservation_rate = len(set(chunks_numbers) & set(original_numbers)) / len(set(original_numbers))
            metrics['number_preservation_rate'] = number_preservation_rate
        else:
            metrics['number_preservation_rate'] = 1.0
        
        # 2. 块大小一致性
        chunk_sizes = [len(chunk['text']) for chunk in chunks]
        if chunk_sizes:
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            size_variance = sum((size - avg_size) ** 2 for size in chunk_sizes) / len(chunk_sizes)
            size_consistency = max(0, 1 - (size_variance ** 0.5) / avg_size)
            metrics['size_consistency'] = size_consistency
        else:
            metrics['size_consistency'] = 0
        
        # 3. 语义完整性（简化评估）
        complete_sentences = 0
        total_chunks = len(chunks)
        
        for chunk in chunks:
            chunk_text = chunk['text'].strip()
            if self._is_semantically_complete(chunk_text):
                complete_sentences += 1
        
        semantic_completeness = complete_sentences / total_chunks if total_chunks > 0 else 0
        metrics['semantic_completeness'] = semantic_completeness
        
        # 4. 重要信息分布
        critical_chunks = sum(1 for chunk in chunks 
                            if chunk.get('metadata', {}).get('contains_numbers', False) or
                               chunk.get('metadata', {}).get('contains_formulas', False))
        
        if original_numbers or self._contains_formulas(original_text):
            critical_distribution = critical_chunks / total_chunks if total_chunks > 0 else 0
            metrics['critical_info_distribution'] = critical_distribution
        else:
            metrics['critical_info_distribution'] = 1.0
        
        # 5. 计算总分
        weights = {
            'number_preservation_rate': 0.4,
            'semantic_completeness': 0.3,
            'size_consistency': 0.2,
            'critical_info_distribution': 0.1
        }
        
        total_score = sum(metrics[key] * weights[key] for key in weights if key in metrics)
        
        return {
            'score': total_score,
            'metrics': metrics,
            'assessment': self._get_quality_assessment(total_score)
        }
    
    def _extract_numbers(self, text: str) -> List[str]:
        """提取文本中的数字"""
        import re
        patterns = [
            r'\d+\.?\d*%',              # 百分比
            r'\d+\.?\d*[万亿千百十]',    # 中文数字单位
            r'\$\d+\.?\d*',             # 货币
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # 日期
            r'\d+:\d+',                 # 时间
            r'\d+\.?\d*',               # 一般数字
        ]
        
        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            numbers.extend(matches)
        
        return numbers
    
    def _contains_formulas(self, text: str) -> bool:
        """检查文本是否包含公式"""
        import re
        formula_patterns = [
            r'[=+\-*/]',
            r'[∫∑∏√]',
            r'[α-ωΑ-Ω]',
        ]
        
        for pattern in formula_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _is_semantically_complete(self, text: str) -> bool:
        """简单检查文本语义是否完整"""
        text = text.strip()
        if not text:
            return False
        
        # 检查是否以句号、问号、感叹号结尾
        sentence_endings = ['.', '!', '?', '。', '！', '？']
        if text[-1] in sentence_endings:
            return True
        
        # 检查是否包含完整的句子结构（简化版）
        if len(text) > 50 and ('。' in text or '. ' in text):
            return True
        
        return False
    
    def _get_quality_assessment(self, score: float) -> str:
        """根据分数获取质量评估"""
        if score >= 0.9:
            return "优秀"
        elif score >= 0.8:
            return "良好"
        elif score >= 0.7:
            return "一般"
        elif score >= 0.6:
            return "需要改进"
        else:
            return "较差" 