"""
嵌入驱动的语义分块器
通过计算句子级嵌入相似性来确定最佳分块边界
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
import asyncio
import aiohttp
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)

@dataclass
class SemanticChunk:
    """语义分块结果"""
    text: str
    start_position: int
    end_position: int
    semantic_coherence_score: float
    contains_critical_info: bool
    metadata: Dict[str, Any]

@dataclass
class SentenceInfo:
    """句子信息"""
    text: str
    start_pos: int
    end_pos: int
    embedding: Optional[np.ndarray] = None
    importance_score: float = 0.0
    contains_numbers: bool = False
    contains_entities: bool = False

class EmbeddingBasedSemanticChunker:
    """基于嵌入的语义分块器"""
    
    def __init__(self, 
                 vector_service_url: str = "http://localhost:8002",
                 coherence_threshold: float = 0.7,
                 min_chunk_size: int = 200,
                 max_chunk_size: int = 1500,
                 critical_info_protection: bool = True):
        """
        初始化语义分块器
        
        Args:
            vector_service_url: 向量服务地址
            coherence_threshold: 语义连贯性阈值
            min_chunk_size: 最小分块大小
            max_chunk_size: 最大分块大小
            critical_info_protection: 是否保护关键信息
        """
        self.vector_service_url = vector_service_url
        self.coherence_threshold = coherence_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.critical_info_protection = critical_info_protection
        
        # 重要信息模式
        self.critical_patterns = [
            r'\d+\.?\d*[%万亿千百十⭐]',    # 数字和单位
            r'\$\d+\.?\d*',                # 货币
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # 日期
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',  # 专有名词
            r'《[^》]+》',                 # 书籍标题
            r'"[^"]*"',                    # 引用内容
        ]
        
        # 句子分割模式（优化中英文混合）
        self.sentence_patterns = [
            r'[.!?。！？]\s+',            # 标准句子结尾
            r'[.!?。！？](?=\n)',         # 换行前的句子结尾
            r'[.!?。！？]$',              # 文本末尾的句子结尾
        ]
    
    async def chunk_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[SemanticChunk]:
        """
        对文档进行语义分块
        
        Args:
            text: 文档文本
            metadata: 文档元数据
            
        Returns:
            语义分块列表
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            return [SemanticChunk(
                text=text,
                start_position=0,
                end_position=len(text),
                semantic_coherence_score=1.0,
                contains_critical_info=self._contains_critical_info(text),
                metadata=metadata or {}
            )]
        
        try:
            # 1. 句子分割和预处理
            sentences = self._split_into_sentences(text)
            logger.info(f"文档分割为 {len(sentences)} 个句子")
            
            # 2. 计算句子嵌入
            sentence_embeddings = await self._compute_sentence_embeddings(sentences)
            
            # 3. 计算语义相似性
            similarity_scores = self._compute_similarity_scores(sentence_embeddings)
            
            # 4. 检测语义边界
            boundaries = self._detect_semantic_boundaries(
                sentences, similarity_scores, text
            )
            
            # 5. 生成语义分块
            chunks = self._create_semantic_chunks(
                text, sentences, boundaries, metadata
            )
            
            logger.info(f"生成 {len(chunks)} 个语义分块")
            return chunks
            
        except Exception as e:
            logger.error(f"语义分块失败: {str(e)}")
            # 降级到简单分块
            return self._fallback_chunking(text, metadata)
    
    def _split_into_sentences(self, text: str) -> List[SentenceInfo]:
        """将文本分割为句子"""
        sentences = []
        current_pos = 0
        
        # 使用多个模式进行句子分割
        sentence_boundaries = []
        
        for pattern in self.sentence_patterns:
            for match in re.finditer(pattern, text):
                sentence_boundaries.append(match.end())
        
        # 去重并排序边界点
        sentence_boundaries = sorted(set(sentence_boundaries))
        sentence_boundaries.insert(0, 0)  # 添加开始位置
        
        for i in range(len(sentence_boundaries) - 1):
            start = sentence_boundaries[i]
            end = sentence_boundaries[i + 1]
            sentence_text = text[start:end].strip()
            
            if sentence_text and len(sentence_text) > 10:  # 过滤过短的句子
                sentence_info = SentenceInfo(
                    text=sentence_text,
                    start_pos=start,
                    end_pos=end,
                    contains_numbers=self._contains_numbers(sentence_text),
                    contains_entities=self._contains_entities(sentence_text),
                    importance_score=self._calculate_importance_score(sentence_text)
                )
                sentences.append(sentence_info)
        
        # 添加最后一部分（如果有）
        if sentence_boundaries and sentence_boundaries[-1] < len(text):
            remaining_text = text[sentence_boundaries[-1]:].strip()
            if remaining_text:
                sentence_info = SentenceInfo(
                    text=remaining_text,
                    start_pos=sentence_boundaries[-1],
                    end_pos=len(text),
                    contains_numbers=self._contains_numbers(remaining_text),
                    contains_entities=self._contains_entities(remaining_text),
                    importance_score=self._calculate_importance_score(remaining_text)
                )
                sentences.append(sentence_info)
        
        return sentences
    
    async def _compute_sentence_embeddings(self, sentences: List[SentenceInfo]) -> np.ndarray:
        """计算句子嵌入"""
        if not sentences:
            return np.array([])
        
        try:
            # 准备文本列表
            texts = [sent.text for sent in sentences]
            
            # 调用向量服务
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.vector_service_url}/vectorize",
                    json={
                        "texts": texts,
                        "store_vectors": False  # 不存储，只获取向量
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        embeddings = np.array(result["vectors"])
                        
                        # 将嵌入保存到句子信息中
                        for i, sentence in enumerate(sentences):
                            if i < len(embeddings):
                                sentence.embedding = embeddings[i]
                        
                        logger.info(f"成功计算 {len(embeddings)} 个句子嵌入")
                        return embeddings
                    else:
                        logger.error(f"向量服务请求失败: {response.status}")
                        return np.array([])
        
        except Exception as e:
            logger.error(f"计算句子嵌入失败: {str(e)}")
            return np.array([])
    
    def _compute_similarity_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """计算相邻句子间的语义相似性"""
        if len(embeddings) < 2:
            return np.array([])
        
        similarities = []
        for i in range(len(embeddings) - 1):
            # 计算相邻句子的余弦相似性
            sim = cosine_similarity(
                embeddings[i].reshape(1, -1),
                embeddings[i + 1].reshape(1, -1)
            )[0, 0]
            similarities.append(sim)
        
        return np.array(similarities)
    
    def _detect_semantic_boundaries(self, 
                                  sentences: List[SentenceInfo], 
                                  similarity_scores: np.ndarray,
                                  full_text: str) -> List[int]:
        """检测语义边界点"""
        if len(similarity_scores) == 0:
            return [0, len(sentences)]
        
        boundaries = [0]  # 总是从0开始
        
        # 方法1: 基于相似性突降检测边界
        # 找到相似性显著下降的点
        threshold = np.mean(similarity_scores) - 0.5 * np.std(similarity_scores)
        low_similarity_points = np.where(similarity_scores < threshold)[0]
        
        # 方法2: 使用峰值检测找到相似性的局部最小值
        # 反转相似性分数，寻找峰值（原来的谷值）
        inverted_scores = 1 - similarity_scores
        peaks, _ = find_peaks(inverted_scores, height=1-self.coherence_threshold)
        
        # 合并两种方法的结果
        potential_boundaries = set(low_similarity_points) | set(peaks)
        
        # 根据文本长度和关键信息调整边界
        adjusted_boundaries = []
        current_chunk_start = 0
        
        for boundary_idx in sorted(potential_boundaries):
            if boundary_idx == 0:
                continue
            
            # 计算当前块的长度
            chunk_start_pos = sentences[current_chunk_start].start_pos
            chunk_end_pos = sentences[boundary_idx].end_pos
            chunk_length = chunk_end_pos - chunk_start_pos
            
            # 检查是否满足最小块大小
            if chunk_length >= self.min_chunk_size:
                # 检查是否有关键信息需要保护
                chunk_text = full_text[chunk_start_pos:chunk_end_pos]
                
                if self.critical_info_protection and self._contains_critical_info(chunk_text):
                    # 如果包含关键信息，尝试在更安全的位置分割
                    safe_boundary = self._find_safe_boundary(
                        sentences, current_chunk_start, boundary_idx, full_text
                    )
                    if safe_boundary is not None:
                        adjusted_boundaries.append(safe_boundary)
                        current_chunk_start = safe_boundary
                    else:
                        adjusted_boundaries.append(boundary_idx)
                        current_chunk_start = boundary_idx
                else:
                    adjusted_boundaries.append(boundary_idx)
                    current_chunk_start = boundary_idx
            
            # 检查是否超过最大块大小
            elif chunk_length > self.max_chunk_size:
                # 强制分割
                forced_boundary = self._force_split_large_chunk(
                    sentences, current_chunk_start, boundary_idx
                )
                adjusted_boundaries.extend(forced_boundary)
                current_chunk_start = forced_boundary[-1] if forced_boundary else boundary_idx
        
        # 添加最终边界
        boundaries.extend(adjusted_boundaries)
        boundaries.append(len(sentences))
        
        # 去重并排序
        boundaries = sorted(set(boundaries))
        
        logger.info(f"检测到 {len(boundaries)-1} 个语义边界")
        return boundaries
    
    def _find_safe_boundary(self, 
                           sentences: List[SentenceInfo], 
                           start_idx: int, 
                           end_idx: int,
                           full_text: str) -> Optional[int]:
        """在关键信息附近找到安全的分割边界"""
        # 检查每个可能的分割点
        for i in range(start_idx + 1, end_idx):
            # 检查分割后两部分是否都保持关键信息完整性
            part1_text = full_text[
                sentences[start_idx].start_pos:sentences[i].end_pos
            ]
            part2_text = full_text[
                sentences[i].start_pos:sentences[end_idx].end_pos
            ]
            
            # 简单检查：确保数字信息没有被分割
            if (not self._has_incomplete_number_info(part1_text) and 
                not self._has_incomplete_number_info(part2_text)):
                return i
        
        return None
    
    def _force_split_large_chunk(self, 
                                sentences: List[SentenceInfo], 
                                start_idx: int, 
                                end_idx: int) -> List[int]:
        """强制分割过大的块"""
        boundaries = []
        current_length = 0
        current_start = start_idx
        
        for i in range(start_idx, end_idx):
            sentence_length = len(sentences[i].text)
            
            if current_length + sentence_length > self.max_chunk_size:
                if i > current_start:  # 确保至少有一个句子
                    boundaries.append(i)
                    current_start = i
                    current_length = sentence_length
                else:
                    # 单个句子就超过最大大小，强制分割
                    current_length += sentence_length
            else:
                current_length += sentence_length
        
        return boundaries
    
    def _create_semantic_chunks(self, 
                               full_text: str, 
                               sentences: List[SentenceInfo], 
                               boundaries: List[int],
                               metadata: Optional[Dict[str, Any]]) -> List[SemanticChunk]:
        """根据边界创建语义分块"""
        chunks = []
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            if start_idx >= len(sentences) or end_idx > len(sentences):
                continue
            
            # 获取分块文本
            chunk_start_pos = sentences[start_idx].start_pos
            chunk_end_pos = sentences[end_idx - 1].end_pos
            chunk_text = full_text[chunk_start_pos:chunk_end_pos].strip()
            
            if not chunk_text:
                continue
            
            # 计算语义连贯性分数
            coherence_score = self._calculate_chunk_coherence(
                sentences[start_idx:end_idx]
            )
            
            # 检查是否包含关键信息
            contains_critical = self._contains_critical_info(chunk_text)
            
            # 构建分块元数据
            chunk_metadata = (metadata or {}).copy()
            chunk_metadata.update({
                "sentence_count": end_idx - start_idx,
                "semantic_coherence": coherence_score,
                "contains_numbers": any(s.contains_numbers for s in sentences[start_idx:end_idx]),
                "contains_entities": any(s.contains_entities for s in sentences[start_idx:end_idx]),
                "avg_importance": np.mean([s.importance_score for s in sentences[start_idx:end_idx]]),
                "chunk_method": "semantic_embedding"
            })
            
            chunk = SemanticChunk(
                text=chunk_text,
                start_position=chunk_start_pos,
                end_position=chunk_end_pos,
                semantic_coherence_score=coherence_score,
                contains_critical_info=contains_critical,
                metadata=chunk_metadata
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _calculate_chunk_coherence(self, sentences: List[SentenceInfo]) -> float:
        """计算分块内部的语义连贯性"""
        if len(sentences) < 2:
            return 1.0
        
        # 如果有嵌入，计算平均相似性
        embeddings = [s.embedding for s in sentences if s.embedding is not None]
        if len(embeddings) >= 2:
            embeddings = np.array(embeddings)
            similarities = []
            
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity(
                    embeddings[i].reshape(1, -1),
                    embeddings[i + 1].reshape(1, -1)
                )[0, 0]
                similarities.append(sim)
            
            return float(np.mean(similarities))
        
        # 降级到基于重要性的连贯性评估
        importance_scores = [s.importance_score for s in sentences]
        importance_variance = np.var(importance_scores)
        
        # 重要性越一致，连贯性越高
        return max(0.0, 1.0 - importance_variance)
    
    def _contains_critical_info(self, text: str) -> bool:
        """检查文本是否包含关键信息"""
        for pattern in self.critical_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _contains_numbers(self, text: str) -> bool:
        """检查文本是否包含数字"""
        return bool(re.search(r'\d+', text))
    
    def _contains_entities(self, text: str) -> bool:
        """检查文本是否包含实体（简化版）"""
        # 检查专有名词模式
        patterns = [
            r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',  # 英文专有名词
            r'[A-Z]{2,}',                        # 缩写词
            r'《[^》]+》',                       # 书籍标题
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _calculate_importance_score(self, text: str) -> float:
        """计算句子重要性分数"""
        score = 0.0
        
        # 基于长度的分数（适中长度得分较高）
        length_score = min(len(text) / 100, 1.0)
        score += length_score * 0.2
        
        # 基于关键信息的分数
        if self._contains_numbers(text):
            score += 0.3
        
        if self._contains_entities(text):
            score += 0.2
        
        # 基于特殊标记的分数
        if any(marker in text for marker in ['重要', '关键', '注意', '必须', '应该']):
            score += 0.2
        
        # 基于问号和感叹号的分数（通常表示重要信息）
        if any(marker in text for marker in ['？', '！', '?', '!']):
            score += 0.1
        
        return min(score, 1.0)
    
    def _has_incomplete_number_info(self, text: str) -> bool:
        """检查文本是否有不完整的数字信息"""
        # 简化检查：查看是否有数字但缺少上下文
        numbers = re.findall(r'\d+\.?\d*', text)
        if not numbers:
            return False
        
        # 检查数字是否有适当的上下文（单位、描述等）
        for number in numbers:
            number_pos = text.find(number)
            
            # 检查数字前后的上下文
            context_before = text[max(0, number_pos-20):number_pos]
            context_after = text[number_pos+len(number):number_pos+len(number)+20]
            
            # 如果数字前后都没有明显的上下文，可能是不完整的
            if (not re.search(r'[A-Za-z\u4e00-\u9fff]', context_before) and 
                not re.search(r'[A-Za-z\u4e00-\u9fff%万亿千百十]', context_after)):
                return True
        
        return False
    
    def _fallback_chunking(self, text: str, metadata: Optional[Dict[str, Any]]) -> List[SemanticChunk]:
        """降级分块策略"""
        logger.info("使用降级分块策略")
        
        # 简单按长度分块
        chunks = []
        chunk_size = min(self.max_chunk_size, max(self.min_chunk_size, len(text) // 3))
        
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i + chunk_size]
            
            chunk = SemanticChunk(
                text=chunk_text,
                start_position=i,
                end_position=min(i + chunk_size, len(text)),
                semantic_coherence_score=0.5,  # 中等连贯性
                contains_critical_info=self._contains_critical_info(chunk_text),
                metadata=(metadata or {}).copy()
            )
            chunks.append(chunk)
        
        return chunks

class ContextAwareOverlapGenerator:
    """上下文感知的重叠生成器"""
    
    def __init__(self, min_overlap: int = 50, max_overlap: int = 300):
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
    
    def generate_overlap(self, 
                        current_chunk: SemanticChunk, 
                        next_chunk: SemanticChunk,
                        full_text: str) -> str:
        """
        生成两个分块之间的上下文感知重叠
        
        Args:
            current_chunk: 当前分块
            next_chunk: 下一个分块
            full_text: 完整文本
            
        Returns:
            重叠内容
        """
        # 从当前分块末尾开始向前查找合适的重叠边界
        current_end = current_chunk.end_position
        next_start = next_chunk.start_position
        
        # 如果两个分块相邻，生成重叠
        if current_end == next_start:
            overlap_start = max(
                current_end - self.max_overlap,
                current_chunk.start_position
            )
            
            overlap_text = full_text[overlap_start:current_end]
            
            # 尝试在句子边界截断
            sentence_boundary = self._find_sentence_boundary(overlap_text)
            if sentence_boundary:
                overlap_text = overlap_text[sentence_boundary:]
            
            # 确保最小重叠长度
            if len(overlap_text) < self.min_overlap and overlap_start > current_chunk.start_position:
                overlap_start = max(
                    current_end - self.min_overlap,
                    current_chunk.start_position
                )
                overlap_text = full_text[overlap_start:current_end]
            
            return overlap_text
        
        return ""
    
    def _find_sentence_boundary(self, text: str) -> Optional[int]:
        """在文本中找到句子边界"""
        sentence_ends = ['.', '!', '?', '。', '！', '？']
        
        for i, char in enumerate(text):
            if char in sentence_ends and i < len(text) - 1:
                # 确保不是缩写中的点
                if char == '.' and i > 0 and text[i-1].isupper():
                    continue
                return i + 1
        
        return None