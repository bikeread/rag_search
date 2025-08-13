"""
分块质量评估器
提供多维度的分块质量评估，支持持续优化
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
from collections import Counter
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """质量评估指标"""
    semantic_coherence: float          # 语义连贯性
    information_preservation: float    # 信息保留度
    size_consistency: float           # 大小一致性
    boundary_quality: float           # 边界质量
    retrieval_effectiveness: float    # 检索有效性
    overall_score: float              # 总体评分
    detailed_metrics: Dict[str, Any]  # 详细指标

@dataclass
class ChunkAnalysis:
    """分块分析结果"""
    chunk_id: str
    text: str
    length: int
    critical_info_density: float
    semantic_completeness: bool
    boundary_score: float
    retrieval_keywords: List[str]
    quality_issues: List[str]

class ChunkQualityAssessor:
    """分块质量评估器"""
    
    def __init__(self, 
                 vector_service_url: str = "http://localhost:8002",
                 reference_corpus: Optional[List[str]] = None):
        """
        初始化质量评估器
        
        Args:
            vector_service_url: 向量服务地址
            reference_corpus: 参考语料库（用于基准对比）
        """
        self.vector_service_url = vector_service_url
        self.reference_corpus = reference_corpus or []
        
        # 质量评估权重
        self.quality_weights = {
            'semantic_coherence': 0.25,
            'information_preservation': 0.30,
            'size_consistency': 0.15,
            'boundary_quality': 0.20,
            'retrieval_effectiveness': 0.10
        }
        
        # 关键信息模式
        self.critical_patterns = {
            'numbers': r'\d+\.?\d*[%万亿千百十⭐]?',
            'dates': r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
            'money': r'\$\d+\.?\d*|¥\d+\.?\d*',
            'entities': r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
            'formulas': r'[=+\-*/]|[∫∑∏√]|[α-ωΑ-Ω]',
            'quotes': r'"[^"]*"|《[^》]+》',
            'technical_terms': r'[A-Z]{2,}|API|HTTP|JSON|XML'
        }
    
    async def assess_chunking_quality(self, 
                                    original_text: str,
                                    chunks: List[Dict[str, Any]],
                                    chunk_method: str = "unknown") -> QualityMetrics:
        """
        评估分块质量
        
        Args:
            original_text: 原始文本
            chunks: 分块列表
            chunk_method: 分块方法
            
        Returns:
            质量评估结果
        """
        logger.info(f"开始评估分块质量，共 {len(chunks)} 个分块")
        
        # 1. 分析每个分块
        chunk_analyses = []
        for i, chunk in enumerate(chunks):
            analysis = await self._analyze_chunk(
                chunk_id=f"chunk_{i}",
                chunk_text=chunk.get('text', ''),
                chunk_metadata=chunk.get('metadata', {})
            )
            chunk_analyses.append(analysis)
        
        # 2. 计算各维度质量指标
        semantic_coherence = await self._assess_semantic_coherence(chunks)
        information_preservation = self._assess_information_preservation(
            original_text, chunks
        )
        size_consistency = self._assess_size_consistency(chunk_analyses)
        boundary_quality = self._assess_boundary_quality(
            original_text, chunk_analyses
        )
        retrieval_effectiveness = await self._assess_retrieval_effectiveness(
            chunk_analyses
        )
        
        # 3. 计算总体评分
        overall_score = (
            semantic_coherence * self.quality_weights['semantic_coherence'] +
            information_preservation * self.quality_weights['information_preservation'] +
            size_consistency * self.quality_weights['size_consistency'] +
            boundary_quality * self.quality_weights['boundary_quality'] +
            retrieval_effectiveness * self.quality_weights['retrieval_effectiveness']
        )
        
        # 4. 生成详细指标
        detailed_metrics = {
            'chunk_count': len(chunks),
            'avg_chunk_length': np.mean([a.length for a in chunk_analyses]),
            'std_chunk_length': np.std([a.length for a in chunk_analyses]),
            'critical_info_chunks': sum(1 for a in chunk_analyses if a.critical_info_density > 0.5),
            'complete_chunks': sum(1 for a in chunk_analyses if a.semantic_completeness),
            'quality_issues': self._aggregate_quality_issues(chunk_analyses),
            'method': chunk_method,
            'assessment_timestamp': asyncio.get_event_loop().time()
        }
        
        logger.info(f"质量评估完成，总分: {overall_score:.3f}")
        
        return QualityMetrics(
            semantic_coherence=semantic_coherence,
            information_preservation=information_preservation,
            size_consistency=size_consistency,
            boundary_quality=boundary_quality,
            retrieval_effectiveness=retrieval_effectiveness,
            overall_score=overall_score,
            detailed_metrics=detailed_metrics
        )
    
    async def _analyze_chunk(self, 
                           chunk_id: str, 
                           chunk_text: str,
                           chunk_metadata: Dict[str, Any]) -> ChunkAnalysis:
        """分析单个分块"""
        # 基础统计
        length = len(chunk_text)
        
        # 关键信息密度
        critical_info_density = self._calculate_critical_info_density(chunk_text)
        
        # 语义完整性
        semantic_completeness = self._check_semantic_completeness(chunk_text)
        
        # 边界质量
        boundary_score = self._assess_chunk_boundary_quality(chunk_text)
        
        # 检索关键词
        retrieval_keywords = self._extract_retrieval_keywords(chunk_text)
        
        # 质量问题
        quality_issues = self._identify_quality_issues(chunk_text, chunk_metadata)
        
        return ChunkAnalysis(
            chunk_id=chunk_id,
            text=chunk_text,
            length=length,
            critical_info_density=critical_info_density,
            semantic_completeness=semantic_completeness,
            boundary_score=boundary_score,
            retrieval_keywords=retrieval_keywords,
            quality_issues=quality_issues
        )
    
    async def _assess_semantic_coherence(self, chunks: List[Dict[str, Any]]) -> float:
        """评估语义连贯性"""
        if len(chunks) < 2:
            return 1.0
        
        try:
            # 获取所有分块的嵌入
            texts = [chunk.get('text', '') for chunk in chunks]
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.vector_service_url}/vectorize",
                    json={"texts": texts, "store_vectors": False}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        embeddings = np.array(result["vectors"])
                        
                        # 计算相邻分块的语义相似性
                        coherence_scores = []
                        for i in range(len(embeddings) - 1):
                            from sklearn.metrics.pairwise import cosine_similarity
                            sim = cosine_similarity(
                                embeddings[i].reshape(1, -1),
                                embeddings[i + 1].reshape(1, -1)
                            )[0, 0]
                            coherence_scores.append(sim)
                        
                        # 返回平均语义连贯性
                        avg_coherence = np.mean(coherence_scores)
                        logger.info(f"语义连贯性评分: {avg_coherence:.3f}")
                        return float(avg_coherence)
        
        except Exception as e:
            logger.error(f"计算语义连贯性失败: {str(e)}")
        
        # 降级到基于文本特征的连贯性评估
        return self._fallback_coherence_assessment(chunks)
    
    def _assess_information_preservation(self, 
                                       original_text: str, 
                                       chunks: List[Dict[str, Any]]) -> float:
        """评估信息保留度"""
        # 提取原始文本的关键信息
        original_info = self._extract_all_critical_info(original_text)
        
        # 提取分块中的关键信息
        chunks_info = {}
        for chunk in chunks:
            chunk_text = chunk.get('text', '')
            chunk_info = self._extract_all_critical_info(chunk_text)
            for info_type, items in chunk_info.items():
                if info_type not in chunks_info:
                    chunks_info[info_type] = set()
                chunks_info[info_type].update(items)
        
        # 计算各类信息的保留率
        preservation_scores = []
        for info_type, original_items in original_info.items():
            if original_items:
                preserved_items = chunks_info.get(info_type, set())
                preservation_rate = len(original_items & preserved_items) / len(original_items)
                preservation_scores.append(preservation_rate)
        
        if preservation_scores:
            avg_preservation = np.mean(preservation_scores)
            logger.info(f"信息保留度评分: {avg_preservation:.3f}")
            return avg_preservation
        
        return 1.0  # 如果没有关键信息，认为完全保留
    
    def _assess_size_consistency(self, analyses: List[ChunkAnalysis]) -> float:
        """评估大小一致性"""
        if len(analyses) < 2:
            return 1.0
        
        lengths = [a.length for a in analyses]
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        
        if mean_length == 0:
            return 0.0
        
        # 变异系数 (CV) 越小，一致性越好
        cv = std_length / mean_length
        consistency_score = max(0.0, 1.0 - cv)
        
        logger.info(f"大小一致性评分: {consistency_score:.3f} (CV: {cv:.3f})")
        return consistency_score
    
    def _assess_boundary_quality(self, 
                                original_text: str, 
                                analyses: List[ChunkAnalysis]) -> float:
        """评估边界质量"""
        if len(analyses) < 2:
            return 1.0
        
        boundary_scores = [a.boundary_score for a in analyses]
        avg_boundary_score = np.mean(boundary_scores)
        
        # 检查是否有信息跨边界分割的情况
        cross_boundary_penalty = self._calculate_cross_boundary_penalty(
            original_text, analyses
        )
        
        final_score = avg_boundary_score * (1.0 - cross_boundary_penalty)
        logger.info(f"边界质量评分: {final_score:.3f}")
        return final_score
    
    async def _assess_retrieval_effectiveness(self, analyses: List[ChunkAnalysis]) -> float:
        """评估检索有效性"""
        # 基于关键词多样性和质量评估检索有效性
        all_keywords = []
        for analysis in analyses:
            all_keywords.extend(analysis.retrieval_keywords)
        
        if not all_keywords:
            return 0.5  # 中等评分
        
        # 计算关键词多样性
        keyword_diversity = len(set(all_keywords)) / len(all_keywords)
        
        # 计算平均关键词质量（基于长度和复杂性）
        keyword_quality_scores = []
        for keyword in set(all_keywords):
            quality_score = min(len(keyword) / 10, 1.0)  # 长度因子
            if re.search(r'[A-Z]', keyword):  # 包含大写字母
                quality_score += 0.1
            if re.search(r'\d', keyword):  # 包含数字
                quality_score += 0.1
            keyword_quality_scores.append(min(quality_score, 1.0))
        
        avg_keyword_quality = np.mean(keyword_quality_scores) if keyword_quality_scores else 0.5
        
        effectiveness_score = (keyword_diversity + avg_keyword_quality) / 2
        logger.info(f"检索有效性评分: {effectiveness_score:.3f}")
        return effectiveness_score
    
    def _calculate_critical_info_density(self, text: str) -> float:
        """计算关键信息密度"""
        if not text:
            return 0.0
        
        critical_matches = 0
        for pattern in self.critical_patterns.values():
            matches = re.findall(pattern, text)
            critical_matches += len(matches)
        
        # 密度 = 关键信息数量 / 文本长度 * 1000 (归一化)
        density = critical_matches / len(text) * 1000
        return min(density, 1.0)
    
    def _check_semantic_completeness(self, text: str) -> bool:
        """检查语义完整性"""
        text = text.strip()
        if not text:
            return False
        
        # 检查句子完整性
        sentence_endings = ['.', '!', '?', '。', '！', '？']
        if text[-1] in sentence_endings:
            return True
        
        # 检查段落完整性
        if '\n\n' in text:
            return True
        
        # 检查是否包含完整的语句结构
        if len(text) > 100 and any(ending in text for ending in sentence_endings):
            return True
        
        return False
    
    def _assess_chunk_boundary_quality(self, chunk_text: str) -> float:
        """评估单个分块的边界质量"""
        score = 0.0
        
        # 开始边界质量
        start_score = 0.5  # 默认分数
        if chunk_text.strip():
            first_char = chunk_text.strip()[0]
            if first_char.isupper() or first_char in '《"\'':  # 以大写字母或引号开始
                start_score = 1.0
            elif first_char.islower():  # 以小写字母开始（可能是分割问题）
                start_score = 0.3
        
        # 结束边界质量
        end_score = 0.5  # 默认分数
        if chunk_text.strip():
            last_char = chunk_text.strip()[-1]
            if last_char in '.!?。！？':  # 以句号结尾
                end_score = 1.0
            elif last_char in ',;，；':  # 以逗号分号结尾（不太理想）
                end_score = 0.6
        
        score = (start_score + end_score) / 2
        return score
    
    def _extract_retrieval_keywords(self, text: str) -> List[str]:
        """提取检索关键词"""
        keywords = []
        
        # 提取专有名词
        proper_nouns = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
        keywords.extend(proper_nouns)
        
        # 提取数字信息
        numbers = re.findall(r'\d+\.?\d*[%万亿千百十⭐]?', text)
        keywords.extend(numbers)
        
        # 提取技术术语
        technical_terms = re.findall(r'[A-Z]{2,}', text)
        keywords.extend(technical_terms)
        
        # 去重并过滤短词
        keywords = list(set([kw for kw in keywords if len(kw) > 2]))
        return keywords[:10]  # 限制数量
    
    def _identify_quality_issues(self, 
                                chunk_text: str, 
                                chunk_metadata: Dict[str, Any]) -> List[str]:
        """识别质量问题"""
        issues = []
        
        # 检查长度问题
        if len(chunk_text) < 50:
            issues.append("chunk_too_short")
        elif len(chunk_text) > 2000:
            issues.append("chunk_too_long")
        
        # 检查边界问题
        text = chunk_text.strip()
        if text and text[0].islower() and not text[0].isdigit():
            issues.append("poor_start_boundary")
        
        if text and text[-1] not in '.!?。！？' and len(text) > 100:
            issues.append("poor_end_boundary")
        
        # 检查内容问题
        if not text:
            issues.append("empty_chunk")
        
        # 检查不完整的关键信息
        if self._has_incomplete_critical_info(text):
            issues.append("incomplete_critical_info")
        
        return issues
    
    def _extract_all_critical_info(self, text: str) -> Dict[str, set]:
        """提取所有关键信息"""
        info = {}
        
        for info_type, pattern in self.critical_patterns.items():
            matches = re.findall(pattern, text)
            info[info_type] = set(matches)
        
        return info
    
    def _fallback_coherence_assessment(self, chunks: List[Dict[str, Any]]) -> float:
        """降级的连贯性评估（基于文本特征）"""
        if len(chunks) < 2:
            return 1.0
        
        coherence_scores = []
        
        for i in range(len(chunks) - 1):
            current_text = chunks[i].get('text', '')
            next_text = chunks[i + 1].get('text', '')
            
            # 基于词汇重叠计算相似性
            current_words = set(re.findall(r'\w+', current_text.lower()))
            next_words = set(re.findall(r'\w+', next_text.lower()))
            
            if current_words or next_words:
                overlap = len(current_words & next_words)
                union = len(current_words | next_words)
                similarity = overlap / union if union > 0 else 0
                coherence_scores.append(similarity)
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
    
    def _calculate_cross_boundary_penalty(self, 
                                        original_text: str, 
                                        analyses: List[ChunkAnalysis]) -> float:
        """计算跨边界分割的惩罚"""
        penalty = 0.0
        
        # 检查关键信息是否被分割
        for i in range(len(analyses) - 1):
            current_chunk = analyses[i]
            next_chunk = analyses[i + 1]
            
            # 检查边界附近是否有被分割的关键信息
            boundary_text = current_chunk.text[-50:] + next_chunk.text[:50]
            
            # 检查数字信息分割
            if self._has_split_numbers(boundary_text):
                penalty += 0.2
            
            # 检查实体分割
            if self._has_split_entities(boundary_text):
                penalty += 0.1
        
        return min(penalty, 1.0)
    
    def _has_incomplete_critical_info(self, text: str) -> bool:
        """检查是否有不完整的关键信息"""
        # 检查数字但缺少单位
        numbers = re.findall(r'\d+\.?\d*', text)
        for number in numbers:
            # 查找数字周围的上下文
            number_pos = text.find(number)
            context = text[max(0, number_pos-10):number_pos+len(number)+10]
            
            # 如果数字没有明显的单位或描述，可能是不完整的
            if not re.search(r'[%万亿千百十$¥]|年|月|日|个|次|倍', context):
                return True
        
        return False
    
    def _has_split_numbers(self, boundary_text: str) -> bool:
        """检查是否有分割的数字"""
        # 简化检查：查找可能被分割的数字模式
        patterns = [
            r'\d+\s*$',  # 结尾的数字
            r'^\s*[%万亿千百十]',  # 开头的单位
            r'\$\s*$',  # 结尾的货币符号
            r'^\s*\d+',  # 开头的数字
        ]
        
        for pattern in patterns:
            if re.search(pattern, boundary_text):
                return True
        
        return False
    
    def _has_split_entities(self, boundary_text: str) -> bool:
        """检查是否有分割的实体"""
        # 检查可能被分割的专有名词
        if re.search(r'[A-Z][a-z]*\s*$', boundary_text) and re.search(r'^\s*[A-Z][a-z]*', boundary_text):
            return True
        
        return False
    
    def _aggregate_quality_issues(self, analyses: List[ChunkAnalysis]) -> Dict[str, int]:
        """聚合质量问题"""
        issue_counts = Counter()
        
        for analysis in analyses:
            for issue in analysis.quality_issues:
                issue_counts[issue] += 1
        
        return dict(issue_counts)

class ChunkQualityReporter:
    """分块质量报告器"""
    
    def __init__(self):
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.8,
            'acceptable': 0.7,
            'poor': 0.6
        }
    
    def generate_quality_report(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """生成质量报告"""
        # 确定质量等级
        quality_level = self._determine_quality_level(metrics.overall_score)
        
        # 生成建议
        recommendations = self._generate_recommendations(metrics)
        
        # 生成报告
        report = {
            "overall_assessment": {
                "score": round(metrics.overall_score, 3),
                "level": quality_level,
                "summary": self._generate_summary(metrics, quality_level)
            },
            "dimension_scores": {
                "semantic_coherence": round(metrics.semantic_coherence, 3),
                "information_preservation": round(metrics.information_preservation, 3),
                "size_consistency": round(metrics.size_consistency, 3),
                "boundary_quality": round(metrics.boundary_quality, 3),
                "retrieval_effectiveness": round(metrics.retrieval_effectiveness, 3)
            },
            "detailed_metrics": metrics.detailed_metrics,
            "recommendations": recommendations,
            "improvement_priority": self._prioritize_improvements(metrics)
        }
        
        return report
    
    def _determine_quality_level(self, score: float) -> str:
        """确定质量等级"""
        for level, threshold in self.quality_thresholds.items():
            if score >= threshold:
                return level
        return "needs_improvement"
    
    def _generate_summary(self, metrics: QualityMetrics, level: str) -> str:
        """生成评估摘要"""
        summaries = {
            'excellent': "分块质量优秀，各项指标表现出色",
            'good': "分块质量良好，大部分指标达到预期",
            'acceptable': "分块质量可接受，存在改进空间",
            'poor': "分块质量较差，需要重点优化",
            'needs_improvement': "分块质量需要显著改进"
        }
        
        base_summary = summaries.get(level, "质量评估完成")
        
        # 添加具体问题
        weak_points = []
        if metrics.semantic_coherence < 0.7:
            weak_points.append("语义连贯性")
        if metrics.information_preservation < 0.8:
            weak_points.append("信息保留度")
        if metrics.boundary_quality < 0.7:
            weak_points.append("边界质量")
        
        if weak_points:
            base_summary += f"，主要问题：{', '.join(weak_points)}"
        
        return base_summary
    
    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if metrics.semantic_coherence < 0.7:
            recommendations.append("建议使用语义感知分块策略，提高分块间的语义连贯性")
        
        if metrics.information_preservation < 0.8:
            recommendations.append("建议启用关键信息保护机制，避免重要数据被分割")
        
        if metrics.size_consistency < 0.6:
            recommendations.append("建议调整分块大小参数，提高分块大小的一致性")
        
        if metrics.boundary_quality < 0.7:
            recommendations.append("建议优化边界检测算法，在句子或段落边界进行分割")
        
        if metrics.retrieval_effectiveness < 0.6:
            recommendations.append("建议增强关键词提取和元数据丰富，提升检索效果")
        
        # 总体建议
        if metrics.overall_score < 0.7:
            recommendations.append("建议综合优化分块策略，考虑使用混合方法")
        
        return recommendations
    
    def _prioritize_improvements(self, metrics: QualityMetrics) -> List[str]:
        """确定改进优先级"""
        scores = {
            'information_preservation': metrics.information_preservation,
            'semantic_coherence': metrics.semantic_coherence,
            'boundary_quality': metrics.boundary_quality,
            'size_consistency': metrics.size_consistency,
            'retrieval_effectiveness': metrics.retrieval_effectiveness
        }
        
        # 按分数从低到高排序
        priority_order = sorted(scores.items(), key=lambda x: x[1])
        
        return [item[0] for item in priority_order if item[1] < 0.8]