"""
文档分割策略模块，提供多种文档分割算法。
包含语义感知分块和数字信息保护机制。
"""

import re
import logging
from typing import List, Dict, Any, Callable, Optional, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ImportanceLevel(Enum):
    """信息重要性级别"""
    CRITICAL = "critical"  # 数字、公式、专有名词
    HIGH = "high"         # 标题、关键词
    MEDIUM = "medium"     # 普通句子
    LOW = "low"          # 连接词、填充词

@dataclass
class TextSegment:
    """文本段落信息"""
    text: str
    start_pos: int
    end_pos: int
    importance: ImportanceLevel
    segment_type: str
    contains_numbers: bool = False
    contains_formulas: bool = False
    contains_proper_nouns: bool = False

class SemanticBoundaryDetector:
    """语义边界检测器"""
    
    def __init__(self):
        # 数字和特殊符号模式
        self.number_patterns = [
            r'\d+\.?\d*%',              # 百分比
            r'\d+\.?\d*[万亿千百十]',    # 中文数字单位
            r'\d+\.?\d*[KMGT]?[B]?',    # 技术单位 (KB, MB, GB等)
            r'\$\d+\.?\d*',             # 货币
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # 日期
            r'\d+:\d+',                 # 时间
            r'\d+\.?\d*',               # 一般数字
        ]
        
        # 公式模式 
        self.formula_patterns = [
            r'[a-zA-Z]+\s*[=+\-*/]\s*[a-zA-Z0-9]+',     # 简单公式
            r'\([^)]*[+\-*/][^)]*\)',                    # 括号内的计算
            r'[∫∑∏√]',                                   # 数学符号
            r'[α-ωΑ-Ω]',                                # 希腊字母
        ]
        
        # 专有名词模式
        self.proper_noun_patterns = [
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',         # 英文专有名词
            r'[A-Z]{2,}',                               # 缩写词
            r'《[^》]+》',                              # 书籍、文章标题
            r'"[^"]*"',                                 # 引用内容
        ]
        
        # 段落边界标记
        self.paragraph_separators = ['\n\n', '\r\n\r\n']
        
        # 句子边界标记
        self.sentence_separators = [
            r'[.!?。！？]\s+',
            r'[.!?。！？]$',
            r'[.!?。！？](?=\n)',
        ]
        
        # 标题模式
        self.title_patterns = [
            r'^#{1,6}\s+.+$',                           # Markdown标题
            r'^\d+\.?\s+[^\n]*$',                       # 数字编号标题
            r'^[一二三四五六七八九十]+[、．]\s*[^\n]*$',  # 中文编号
        ]

    def detect_important_segments(self, text: str) -> List[TextSegment]:
        """检测文本中的重要段落"""
        segments = []
        
        # 首先按段落分割
        paragraphs = self._split_by_paragraphs(text)
        
        for para_text, para_start, para_end in paragraphs:
            # 分析段落重要性
            importance, segment_info = self._analyze_paragraph_importance(para_text)
            
            segment = TextSegment(
                text=para_text,
                start_pos=para_start,
                end_pos=para_end,
                importance=importance,
                segment_type=segment_info['type'],
                contains_numbers=segment_info['has_numbers'],
                contains_formulas=segment_info['has_formulas'],
                contains_proper_nouns=segment_info['has_proper_nouns']
            )
            segments.append(segment)
        
        return segments
    
    def _split_by_paragraphs(self, text: str) -> List[tuple]:
        """按段落分割文本"""
        paragraphs = []
        current_pos = 0
        
        # 使用多种段落分隔符
        para_pattern = r'\n\s*\n|\r\n\s*\r\n'
        
        for match in re.finditer(para_pattern, text):
            para_text = text[current_pos:match.start()].strip()
            if para_text:
                paragraphs.append((para_text, current_pos, match.start()))
            current_pos = match.end()
        
        # 添加最后一段
        if current_pos < len(text):
            para_text = text[current_pos:].strip()
            if para_text:
                paragraphs.append((para_text, current_pos, len(text)))
        
        return paragraphs
    
    def _analyze_paragraph_importance(self, text: str) -> tuple:
        """分析段落重要性"""
        info = {
            'type': 'normal',
            'has_numbers': False,
            'has_formulas': False,
            'has_proper_nouns': False
        }
        
        # 检查是否包含数字
        for pattern in self.number_patterns:
            if re.search(pattern, text):
                info['has_numbers'] = True
                break
        
        # 检查是否包含公式
        for pattern in self.formula_patterns:
            if re.search(pattern, text):
                info['has_formulas'] = True
                break
        
        # 检查是否包含专有名词
        for pattern in self.proper_noun_patterns:
            if re.search(pattern, text):
                info['has_proper_nouns'] = True
                break
        
        # 检查是否是标题
        for pattern in self.title_patterns:
            if re.search(pattern, text, re.MULTILINE):
                info['type'] = 'title'
                break
        
        # 根据内容确定重要性级别
        if info['has_numbers'] or info['has_formulas']:
            importance = ImportanceLevel.CRITICAL
        elif info['type'] == 'title' or info['has_proper_nouns']:
            importance = ImportanceLevel.HIGH  
        elif len(text.strip()) > 100:  # 长段落通常包含重要信息
            importance = ImportanceLevel.MEDIUM
        else:
            importance = ImportanceLevel.LOW
        
        return importance, info

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
    
    @staticmethod
    def semantic_aware_split(text: str, chunk_size: int, chunk_overlap: int, 
                           preserve_numbers: bool = True) -> List[Dict[str, Any]]:
        """语义感知分块策略，保护数字信息和重要内容。
        
        Args:
            text: 输入文本
            chunk_size: 块大小
            chunk_overlap: 重叠大小
            preserve_numbers: 是否保护数字信息
            
        Returns:
            分割后的文本块列表，包含元数据
        """
        if not text or not text.strip():
            return []
        
        detector = SemanticBoundaryDetector()
        segments = detector.detect_important_segments(text)
        
        chunks = []
        current_chunk = {
            'text': '',
            'metadata': {
                'importance_levels': set(),
                'contains_numbers': False,
                'contains_formulas': False,
                'contains_proper_nouns': False,
                'segment_types': set()
            }
        }
        
        for segment in segments:
            segment_text = segment.text + ' '  # 添加分隔符
            
            # 如果添加这个段落后超过chunk_size，需要处理当前chunk
            if (len(current_chunk['text']) + len(segment_text)) > chunk_size and current_chunk['text']:
                
                # 如果当前segment是关键信息，尝试调整边界
                if preserve_numbers and segment.importance == ImportanceLevel.CRITICAL:
                    # 检查是否可以通过减少上一个chunk来保持关键信息完整
                    if len(segment_text) <= chunk_size:
                        # 完成当前chunk
                        chunks.append(SplitStrategy._finalize_chunk(current_chunk))
                        
                        # 创建新chunk，将关键segment完整保留
                        current_chunk = {
                            'text': segment_text,
                            'metadata': {
                                'importance_levels': {segment.importance.value},
                                'contains_numbers': segment.contains_numbers,
                                'contains_formulas': segment.contains_formulas,
                                'contains_proper_nouns': segment.contains_proper_nouns,
                                'segment_types': {segment.segment_type}
                            }
                        }
                        continue
                
                # 标准处理：完成当前chunk
                chunks.append(SplitStrategy._finalize_chunk(current_chunk))
                
                # 计算overlap内容
                overlap_text = SplitStrategy._get_overlap_text(current_chunk['text'], chunk_overlap)
                current_chunk = {
                    'text': overlap_text + segment_text,
                    'metadata': {
                        'importance_levels': {segment.importance.value},
                        'contains_numbers': segment.contains_numbers,
                        'contains_formulas': segment.contains_formulas,
                        'contains_proper_nouns': segment.contains_proper_nouns,
                        'segment_types': {segment.segment_type}
                    }
                }
            else:
                # 添加到当前chunk
                current_chunk['text'] += segment_text
                current_chunk['metadata']['importance_levels'].add(segment.importance.value)
                current_chunk['metadata']['contains_numbers'] |= segment.contains_numbers
                current_chunk['metadata']['contains_formulas'] |= segment.contains_formulas
                current_chunk['metadata']['contains_proper_nouns'] |= segment.contains_proper_nouns
                current_chunk['metadata']['segment_types'].add(segment.segment_type)
        
        # 添加最后一个chunk
        if current_chunk['text'].strip():
            chunks.append(SplitStrategy._finalize_chunk(current_chunk))
        
        # 如果没有生成任何chunk，使用fallback策略
        if not chunks:
            logger.warning("语义感知分块没有产生结果，使用fallback策略")
            fallback_chunks = SplitStrategy.by_character(text, chunk_size, chunk_overlap)
            chunks = [{'text': chunk, 'metadata': {'split_method': 'fallback'}} for chunk in fallback_chunks]
        
        logger.info(f"语义感知分块完成：{len(text)} 字符 -> {len(chunks)} 个块")
        for i, chunk in enumerate(chunks):
            logger.debug(f"Chunk {i}: {len(chunk['text'])} 字符, "
                        f"重要性级别: {chunk['metadata'].get('importance_levels', [])}")
        
        return chunks
    
    @staticmethod
    def _finalize_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
        """完成chunk的元数据处理"""
        # 转换set为list以支持JSON序列化
        metadata = chunk['metadata'].copy()
        metadata['importance_levels'] = list(metadata['importance_levels'])
        metadata['segment_types'] = list(metadata['segment_types'])
        metadata['chunk_length'] = len(chunk['text'])
        metadata['split_method'] = 'semantic_aware'
        
        return {
            'text': chunk['text'].strip(),
            'metadata': metadata
        }
    
    @staticmethod
    def _get_overlap_text(text: str, overlap_size: int) -> str:
        """获取用于重叠的文本"""
        if len(text) <= overlap_size:
            return text
        
        # 尽量在句子边界截取
        overlap_text = text[-overlap_size:]
        
        # 寻找句子边界
        sentence_ends = ['.', '!', '?', '。', '！', '？']
        for i, char in enumerate(overlap_text):
            if char in sentence_ends and i < len(overlap_text) - 1:
                return overlap_text[i+1:].strip()
        
        return overlap_text.strip()
    
    @staticmethod
    def adaptive_split(text: str, target_chunk_size: int = 1000, 
                      max_chunk_size: int = 1500, min_chunk_size: int = 500,
                      overlap_ratio: float = 0.2) -> List[Dict[str, Any]]:
        """自适应分块策略，根据内容特点调整分块参数。
        
        Args:
            text: 输入文本
            target_chunk_size: 目标块大小
            max_chunk_size: 最大块大小
            min_chunk_size: 最小块大小
            overlap_ratio: 重叠比例
            
        Returns:
            分割后的文本块列表
        """
        if not text or not text.strip():
            return []
        
        # 分析文本特征
        text_features = SplitStrategy._analyze_text_features(text)
        
        # 根据特征调整参数
        adjusted_chunk_size = target_chunk_size
        adjusted_overlap = int(target_chunk_size * overlap_ratio)
        
        # 如果文本包含大量数字和公式，减少chunk大小以保持精度
        if text_features['number_density'] > 0.1:  # 每100字符超过10个数字
            adjusted_chunk_size = min(target_chunk_size, 800)
            adjusted_overlap = int(adjusted_chunk_size * 0.25)  # 增加重叠
            logger.info("检测到高数字密度，调整为较小分块")
        
        # 如果文本结构化程度高（如技术文档），使用语义感知分块
        if text_features['structure_score'] > 0.7:
            return SplitStrategy.semantic_aware_split(
                text, adjusted_chunk_size, adjusted_overlap, preserve_numbers=True
            )
        
        # 否则使用改进的分隔符策略
        separators = SplitStrategy._get_optimal_separators(text_features)
        chunks = SplitStrategy.by_separator(text, adjusted_chunk_size, adjusted_overlap, separators)
        
        return [{'text': chunk, 'metadata': {'split_method': 'adaptive'}} for chunk in chunks]
    
    @staticmethod
    def _analyze_text_features(text: str) -> Dict[str, float]:
        """分析文本特征"""
        features = {
            'length': len(text),
            'number_density': 0.0,
            'formula_density': 0.0,
            'structure_score': 0.0,
            'language_mix': 0.0
        }
        
        if not text:
            return features
        
        # 数字密度
        number_matches = re.findall(r'\d+\.?\d*', text)
        features['number_density'] = len(number_matches) / len(text) * 100
        
        # 公式密度
        formula_patterns = [r'[=+\-*/]', r'[∫∑∏√]', r'[α-ωΑ-Ω]']
        formula_count = sum(len(re.findall(pattern, text)) for pattern in formula_patterns)
        features['formula_density'] = formula_count / len(text) * 100
        
        # 结构化程度（标题、列表等）
        structure_markers = [
            r'^#{1,6}\s+',          # Markdown标题
            r'^\d+\.?\s+',          # 数字编号
            r'^[•\-*]\s+',          # 列表标记
            r'\n\s*\n',             # 段落分隔
        ]
        structure_count = sum(len(re.findall(pattern, text, re.MULTILINE)) 
                            for pattern in structure_markers)
        features['structure_score'] = min(structure_count / (len(text) / 1000), 1.0)
        
        # 中英文混合程度
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        total_chars = chinese_chars + english_chars
        if total_chars > 0:
            features['language_mix'] = min(chinese_chars, english_chars) / total_chars
        
        return features
    
    @staticmethod
    def _get_optimal_separators(features: Dict[str, float]) -> List[str]:
        """根据文本特征选择最优分隔符"""
        separators = []
        
        # 基础分隔符
        separators.extend(["\n\n", "\n"])
        
        # 根据语言特征选择句子分隔符
        if features.get('language_mix', 0) > 0.3:  # 中英混合
            separators.extend(["。", ". ", "！", "! ", "？", "? "])
        elif features.get('chinese_chars', 0) > features.get('english_chars', 0):  # 主要是中文
            separators.extend(["。", "！", "？", "；", "，"])
        else:  # 主要是英文
            separators.extend([". ", "! ", "? ", "; ", ", "])
        
        # 通用分隔符
        separators.extend([" ", ""])
        
        return separators