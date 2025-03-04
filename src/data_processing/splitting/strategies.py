"""
文档分割策略模块，提供多种文档分割算法。
"""

import re
import logging
from typing import List, Dict, Any, Callable

logger = logging.getLogger(__name__)

class SplitStrategy:
    """文档分割策略类，提供多种分割算法。"""
    
    @staticmethod
    def by_character(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """按字符分割文本。
        
        Args:
            text: 文本内容
            chunk_size: 块大小
            chunk_overlap: 重叠大小
            
        Returns:
            分割后的文本块
        """
        if not text:
            return []
            
        # 如果文本长度小于分块大小，直接返回整个文本
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # 计算结束位置
            end = min(start + chunk_size, len(text))
            
            # 添加分块
            chunks.append(text[start:end])
            
            # 更新开始位置，并考虑重叠
            start = end - chunk_overlap
        
        return chunks
    
    @staticmethod
    def by_separator(text: str, chunk_size: int, chunk_overlap: int, separators: List[str]) -> List[str]:
        """按分隔符分割文本。
        
        Args:
            text: 文本内容
            chunk_size: 块大小
            chunk_overlap: 重叠大小
            separators: 分隔符列表，按优先级排序
            
        Returns:
            分割后的文本块
        """
        if not text:
            return []
            
        # 如果文本长度小于分块大小，直接返回整个文本
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # 计算结束位置
            end = start + chunk_size
            
            # 如果结束位置超出文本长度，直接到结尾
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # 尝试找到合适的分隔符
            found_separator = False
            
            for separator in separators:
                if not separator:
                    continue
                    
                # 找最接近结束位置的分隔符
                separator_position = text.rfind(separator, start, end)
                
                if separator_position != -1:
                    # 分隔符的位置加上分隔符的长度作为实际结束位置
                    actual_end = separator_position + len(separator)
                    chunks.append(text[start:actual_end])
                    start = actual_end - chunk_overlap
                    found_separator = True
                    break
            
            # 如果没有找到合适的分隔符，就强制分割
            if not found_separator:
                chunks.append(text[start:end])
                start = end - chunk_overlap
        
        return chunks
    
    @staticmethod
    def by_regex(text: str, chunk_size: int, chunk_overlap: int, pattern: str) -> List[str]:
        """按正则表达式分割文本。
        
        Args:
            text: 文本内容
            chunk_size: 块大小
            chunk_overlap: 重叠大小
            pattern: 正则表达式模式
            
        Returns:
            分割后的文本块
        """
        if not text:
            return []
            
        # 使用正则表达式分割文本
        segments = re.split(pattern, text)
        segments = [s.strip() for s in segments if s.strip()]
        
        # 如果分割后的段落都很小，可以直接返回
        if all(len(s) <= chunk_size for s in segments):
            return segments
        
        # 否则，重新组合段落
        chunks = []
        current_chunk = ""
        
        for segment in segments:
            # 如果当前段落加上新段落不超过块大小，合并
            if len(current_chunk) + len(segment) <= chunk_size:
                if current_chunk:
                    current_chunk += " "
                current_chunk += segment
            else:
                # 如果当前块非空，添加到结果
                if current_chunk:
                    chunks.append(current_chunk)
                
                # 如果段落本身超过块大小，使用字符分割
                if len(segment) > chunk_size:
                    sub_chunks = SplitStrategy.by_character(segment, chunk_size, chunk_overlap)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    # 否则，新段落作为当前块
                    current_chunk = segment
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    @staticmethod
    def by_paragraph(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """按段落分割文本。
        
        Args:
            text: 文本内容
            chunk_size: 块大小
            chunk_overlap: 重叠大小
            
        Returns:
            分割后的文本块
        """
        return SplitStrategy.by_regex(text, chunk_size, chunk_overlap, r"\n\s*\n")
    
    @staticmethod
    def by_sentence(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """按句子分割文本。
        
        Args:
            text: 文本内容
            chunk_size: 块大小
            chunk_overlap: 重叠大小
            
        Returns:
            分割后的文本块
        """
        return SplitStrategy.by_regex(text, chunk_size, chunk_overlap, r"[.!?。！？]\s+")
    
    @staticmethod
    def for_code(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """针对代码的分割策略。
        
        Args:
            text: 代码文本
            chunk_size: 块大小
            chunk_overlap: 重叠大小
            
        Returns:
            分割后的代码块
        """
        # 使用代码相关的分隔符
        separators = ["\n\n", "\n", ";", "{", "}", "class ", "def ", "function ", "//", "/*", "*/", "#"]
        return SplitStrategy.by_separator(text, chunk_size, chunk_overlap, separators)
    
    @staticmethod
    def for_chinese(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """针对中文的分割策略。
        
        Args:
            text: 中文文本
            chunk_size: 块大小
            chunk_overlap: 重叠大小
            
        Returns:
            分割后的文本块
        """
        # 使用中文相关的分隔符
        separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " "]
        return SplitStrategy.by_separator(text, chunk_size, chunk_overlap, separators)