from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import os
import time
import asyncio
import aiohttp
import json
from datetime import datetime
import numpy as np
from rank_bm25 import BM25Okapi
from collections import defaultdict
import re
import jieba
import logging
from functools import lru_cache
import hashlib

# Cross-Encoder重排序相关导入 - Task3 RAG优化
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
    print("✅ Cross-Encoder依赖加载成功")
except ImportError as e:
    CROSS_ENCODER_AVAILABLE = False
    print(f"⚠️  Cross-Encoder依赖未安装: {e}")
    print("可执行: pip install sentence-transformers torch")

class VectorServiceClient:
    """向量服务客户端"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("VECTOR_SERVICE_URL", "http://localhost:8002")
        
    async def search_vectors(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相似向量"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "query_text": query_text,
                    "top_k": top_k
                }
                
                async with session.post(
                    f"{self.base_url}/search",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("results", [])
                    else:
                        error_text = await response.text()
                        print(f"向量搜索失败: {response.status} - {error_text}")
                        return []
                        
        except Exception as e:
            print(f"调用向量服务失败: {str(e)}")
            return []

class QueryOptimizer:
    """查询优化器 - Task4多轮查询机制的核心组件"""
    
    def __init__(self):
        self.rewrite_patterns = {
            'vague_query': {
                'patterns': [r'^.{1,20}$', r'什么.*？$', r'怎么.*？$', r'能否.*？$'],
                'strategy': 'expand_context'
            },
            'multi_aspect': {
                'patterns': [r'.*和.*', r'.*与.*', r'.*以及.*', r'.*还有.*'],
                'strategy': 'split_aspects'
            },
            'comparison': {
                'patterns': [r'.*区别.*', r'.*对比.*', r'.*差异.*', r'.*优缺点.*'],
                'strategy': 'enhance_comparison'
            },
            'technical_deep': {
                'patterns': [r'.*原理.*', r'.*机制.*', r'.*架构.*', r'.*实现.*'],
                'strategy': 'add_technical_depth'
            }
        }
    
    def assess_query_quality(self, query: str, context: str = "") -> Dict[str, Any]:
        """评估查询质量，决定是否需要多轮查询"""
        assessment = {
            'clarity_score': 0.0,
            'specificity_score': 0.0,
            'completeness_score': 0.0,
            'needs_multiround': False,
            'rewrite_strategy': None,
            'issues': []
        }
        
        # 1. 清晰度评估
        if len(query) < 10:
            assessment['issues'].append('查询过于简短')
            assessment['clarity_score'] = 0.3
        elif len(query) > 200:
            assessment['issues'].append('查询过于冗长') 
            assessment['clarity_score'] = 0.6
        else:
            assessment['clarity_score'] = 0.8
            
        # 2. 具体性评估
        specific_terms = len(re.findall(r'\b(?:具体|详细|准确|精确|确切)\b', query))
        vague_terms = len(re.findall(r'\b(?:大概|大约|可能|似乎|好像)\b', query))
        assessment['specificity_score'] = max(0.3, min(1.0, (specific_terms - vague_terms + 1) * 0.3))
        
        # 3. 完整性评估(基于上下文相关性)
        if context:
            query_words = set(re.findall(r'\w+', query.lower()))
            context_words = set(re.findall(r'\w+', context.lower()))
            relevance = len(query_words & context_words) / len(query_words) if query_words else 0
            assessment['completeness_score'] = relevance
        else:
            assessment['completeness_score'] = 0.5  # 默认值
        
        # 4. 决定是否需要多轮查询
        overall_score = (assessment['clarity_score'] + 
                        assessment['specificity_score'] + 
                        assessment['completeness_score']) / 3
        
        # 提高多轮查询触发阈值，减少不必要的多轮查询
        if overall_score < 0.4 and len(assessment['issues']) > 1:
            assessment['needs_multiround'] = True
            assessment['rewrite_strategy'] = self._determine_rewrite_strategy(query)
        
        return assessment
    
    def _determine_rewrite_strategy(self, query: str) -> str:
        """确定查询重写策略"""
        for strategy, config in self.rewrite_patterns.items():
            for pattern in config['patterns']:
                if re.search(pattern, query, re.IGNORECASE):
                    return config['strategy']
        return 'expand_context'  # 默认策略
    
    def generate_supplementary_queries(self, original_query: str, strategy: str) -> List[str]:
        """生成补充查询"""
        supplementary = []
        
        if strategy == 'expand_context':
            # 为模糊查询生成更具体的版本
            base_keywords = re.findall(r'\b\w{3,}\b', original_query)
            if base_keywords:
                supplementary.extend([
                    f"{base_keywords[0]} 的具体定义和特点",
                    f"{base_keywords[0]} 的应用场景和实例",
                ])
        
        elif strategy == 'split_aspects':
            # 将多方面查询拆分
            connectors = ['和', '与', '以及', '还有']
            for connector in connectors:
                if connector in original_query:
                    parts = original_query.split(connector)
                    if len(parts) >= 2:
                        supplementary.extend([
                            f"{parts[0].strip()} 的特点",
                            f"{parts[1].strip()} 的特点"
                        ])
                    break
        
        elif strategy == 'enhance_comparison':
            # 增强对比查询
            if '区别' in original_query or '差异' in original_query:
                supplementary.extend([
                    original_query.replace('区别', '优势对比'),
                    original_query.replace('区别', '应用场景对比')
                ])
        
        elif strategy == 'add_technical_depth':
            # 为技术查询添加深度
            tech_terms = re.findall(r'\b(?:原理|机制|架构|实现)\b', original_query)
            if tech_terms:
                supplementary.extend([
                    original_query + " 的技术细节",
                    original_query + " 的实际案例"
                ])
        
        return supplementary[:2]  # 最多返回2个补充查询

class MultiRoundRetriever:
    """多轮检索器 - 管理多轮查询的检索和融合"""
    
    def __init__(self, hybrid_retriever, query_optimizer):
        self.hybrid_retriever = hybrid_retriever
        self.query_optimizer = query_optimizer
        self.max_rounds = 2  # 优化：减少最大轮数以3到2
        self.max_supplementary = 1  # 优化：减少每轮补充查询数以2到1
    
    async def multi_round_search(self, original_query: str, top_k: int = 5) -> Dict[str, Any]:
        """执行多轮检索"""
        print("🔄 启动多轮查询机制")
        
        all_results = []
        query_rounds = []
        
        # 第一轮：原始查询
        print(f"📍 第1轮查询：{original_query}")
        round1_results = await self.hybrid_retriever.hybrid_search(original_query, top_k)
        all_results.extend(round1_results)
        query_rounds.append({
            'round': 1,
            'query': original_query,
            'results_count': len(round1_results),
            'type': 'original'
        })
        
        # 评估是否需要补充查询
        context_preview = self._build_context_preview(round1_results)
        quality_assessment = self.query_optimizer.assess_query_quality(original_query, context_preview)
        
        print(f"📊 查询质量评估: 整体分数={quality_assessment.get('clarity_score', 0)*100:.0f}%, "
              f"需要多轮={quality_assessment.get('needs_multiround', False)}")
        
        if quality_assessment['needs_multiround']:
            # 生成补充查询
            strategy = quality_assessment.get('rewrite_strategy', 'expand_context')
            supplementary_queries = self.query_optimizer.generate_supplementary_queries(original_query, strategy)
            
            # 执行补充查询
            for i, sup_query in enumerate(supplementary_queries[:self.max_supplementary]):
                round_num = i + 2
                if round_num > self.max_rounds:
                    break
                    
                print(f"📍 第{round_num}轮查询：{sup_query}")
                round_results = await self.hybrid_retriever.hybrid_search(sup_query, top_k//2)
                all_results.extend(round_results)
                query_rounds.append({
                    'round': round_num,
                    'query': sup_query,
                    'results_count': len(round_results),
                    'type': 'supplementary',
                    'strategy': strategy
                })
        
        # 融合多轮结果
        final_results = self._fuse_multiround_results(all_results, top_k)
        
        return {
            'results': final_results,
            'rounds_executed': len(query_rounds),
            'query_rounds': query_rounds,
            'total_raw_results': len(all_results),
            'final_results_count': len(final_results),
            'multiround_triggered': quality_assessment['needs_multiround']
        }
    
    def _build_context_preview(self, results: List[Dict]) -> str:
        """构建上下文预览用于质量评估"""
        if not results:
            return ""
        
        preview_texts = []
        for result in results[:3]:  # 只取前3个结果
            text = result.get('text', '')[:100]  # 截取前100字符
            if text.strip():
                preview_texts.append(text.strip())
        
        return ' '.join(preview_texts)
    
    def _fuse_multiround_results(self, all_results: List[Dict], top_k: int) -> List[Dict]:
        """融合多轮查询结果"""
        if not all_results:
            return []
        
        # 按文档ID去重，保留最高分数的版本
        doc_scores = {}
        for result in all_results:
            doc_id = result.get('id', str(hash(result.get('text', ''))))
            current_score = result.get('hybrid_score', result.get('score', 0))
            
            if doc_id not in doc_scores or current_score > doc_scores[doc_id]['score']:
                doc_scores[doc_id] = {
                    'result': result,
                    'score': current_score
                }
        
        # 排序并返回Top-K
        sorted_results = sorted(
            doc_scores.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )
        
        final_results = []
        for item in sorted_results[:top_k]:
            result = item['result'].copy()
            result['multiround_fused'] = True
            final_results.append(result)
        
        return final_results

class ContextQualityAssessor:
    """上下文质量评估器"""
    
    def __init__(self):
        self.min_context_length = 100
        self.optimal_context_length = 1500
        self.max_context_length = 4000
    
    def assess_context_quality(self, context: str, query: str, sources: List[Dict]) -> Dict[str, Any]:
        """评估上下文质量"""
        assessment = {
            'length_score': 0.0,
            'relevance_score': 0.0,
            'diversity_score': 0.0,
            'coverage_score': 0.0,
            'needs_enhancement': False,
            'enhancement_suggestions': []
        }
        
        # 1. 长度评分
        context_len = len(context)
        if context_len < self.min_context_length:
            assessment['length_score'] = 0.3
            assessment['enhancement_suggestions'].append('上下文过短，需要更多信息')
        elif context_len > self.max_context_length:
            assessment['length_score'] = 0.7
            assessment['enhancement_suggestions'].append('上下文过长，需要精简')
        else:
            assessment['length_score'] = min(1.0, context_len / self.optimal_context_length)
        
        # 2. 相关性评分
        query_words = set(re.findall(r'\w+', query.lower()))
        context_words = set(re.findall(r'\w+', context.lower()))
        if query_words:
            relevance = len(query_words & context_words) / len(query_words)
            assessment['relevance_score'] = relevance
        
        # 3. 多样性评分（文档来源多样性）
        if sources:
            unique_sources = len(set(src.get('metadata', {}).get('document_id', f'doc_{i}') 
                                   for i, src in enumerate(sources)))
            assessment['diversity_score'] = min(1.0, unique_sources / max(len(sources), 1))
        
        # 4. 覆盖率评分（基于查询关键词覆盖）
        critical_terms = self._extract_critical_terms(query)
        covered_terms = sum(1 for term in critical_terms if term.lower() in context.lower())
        if critical_terms:
            assessment['coverage_score'] = covered_terms / len(critical_terms)
        else:
            assessment['coverage_score'] = 0.8  # 默认较高分数
        
        # 5. 决定是否需要增强
        overall_score = (assessment['length_score'] + assessment['relevance_score'] + 
                        assessment['diversity_score'] + assessment['coverage_score']) / 4
        
        if overall_score < 0.7:
            assessment['needs_enhancement'] = True
        
        return assessment
    
    def _extract_critical_terms(self, query: str) -> List[str]:
        """提取查询中的关键术语"""
        # 提取专业术语、数字、重要名词
        critical_patterns = [
            r'\b\d+\b',  # 数字
            r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b',  # 驼峰命名
            r'\b[a-zA-Z]{4,}\b'  # 较长的单词
        ]
        
        terms = []
        for pattern in critical_patterns:
            terms.extend(re.findall(pattern, query))
        
        # 去重并返回
        return list(set(terms))

# Task5 - 质量控制系统
class AnswerQualityController:
    """答案质量控制器 - Task5核心组件"""
    
    def __init__(self):
        self.quality_thresholds = {
            'min_relevance_score': 0.4,
            'min_completeness_score': 0.5,
            'min_confidence_score': 0.6,
            'min_coherence_score': 0.5,
            'max_regeneration_attempts': 2
        }
        
        # 质量问题模式识别
        self.quality_issues_patterns = {
            'incomplete_answer': [
                r'抱歉.*无法.*',
                r'没有.*相关.*信息',
                r'无法.*确定',
                r'需要.*更多.*信息'
            ],
            'contradictory_info': [
                r'但是.*然而',
                r'相反.*同时',
                r'一方面.*另一方面.*矛盾'
            ],
            'vague_response': [
                r'^.{10,50}$',  # 过短回答
                r'可能.*大概.*或许',
                r'通常.*一般.*往往.*没有具体'
            ],
            'factual_errors': [
                r'年份.*错误',
                r'\d{4}年.*\d{4}年.*矛盾',
                r'版本.*不一致'
            ]
        }
        
        # 逻辑一致性检查器
        self.consistency_checkers = {
            'temporal': self._check_temporal_consistency,
            'numerical': self._check_numerical_consistency, 
            'categorical': self._check_categorical_consistency,
            'causal': self._check_causal_consistency
        }
    
    def comprehensive_quality_assessment(self, 
                                       answer: str, 
                                       query: str, 
                                       context: str, 
                                       sources: List[Dict]) -> Dict[str, Any]:
        """全面的答案质量评估"""
        assessment = {
            'overall_score': 0.0,
            'component_scores': {},
            'quality_issues': [],
            'improvement_suggestions': [],
            'confidence_level': 'medium',
            'requires_regeneration': False
        }
        
        # 1. 相关性评估
        relevance_score = self._assess_relevance(answer, query)
        assessment['component_scores']['relevance'] = relevance_score
        
        # 2. 完整性评估
        completeness_score = self._assess_completeness(answer, query, context)
        assessment['component_scores']['completeness'] = completeness_score
        
        # 3. 准确性评估
        accuracy_score = self._assess_accuracy(answer, sources)
        assessment['component_scores']['accuracy'] = accuracy_score
        
        # 4. 连贯性评估
        coherence_score = self._assess_coherence(answer)
        assessment['component_scores']['coherence'] = coherence_score
        
        # 5. 逻辑一致性检查
        consistency_results = self._check_logical_consistency(answer, context)
        assessment['component_scores']['consistency'] = consistency_results['overall_score']
        assessment['quality_issues'].extend(consistency_results['issues'])
        
        # 6. 质量问题检测
        detected_issues = self._detect_quality_issues(answer)
        assessment['quality_issues'].extend(detected_issues)
        
        # 7. 计算综合分数
        scores = list(assessment['component_scores'].values())
        assessment['overall_score'] = sum(scores) / len(scores) if scores else 0.0
        
        # 8. 决定是否需要重新生成 - 提高阈值减少重试
        assessment['requires_regeneration'] = (
            assessment['overall_score'] < 0.3 or  # 只在质量很差时重试
            len(assessment['quality_issues']) > 3 or  # 增加问题阈值
            any('critical' in issue for issue in assessment['quality_issues'])
        )
        
        # 9. 置信度评估
        if assessment['overall_score'] > 0.8:
            assessment['confidence_level'] = 'high'
        elif assessment['overall_score'] < 0.4:
            assessment['confidence_level'] = 'low'
        else:
            assessment['confidence_level'] = 'medium'
        
        # 10. 生成改进建议
        assessment['improvement_suggestions'] = self._generate_improvement_suggestions(
            assessment['component_scores'], assessment['quality_issues']
        )
        
        return assessment
    
    def _assess_relevance(self, answer: str, query: str) -> float:
        """评估答案相关性"""
        if not answer or not query:
            return 0.0
        
        # 关键词重叠度
        query_words = set(re.findall(r'\w+', query.lower()))
        answer_words = set(re.findall(r'\w+', answer.lower()))
        overlap = len(query_words & answer_words)
        relevance = overlap / len(query_words) if query_words else 0.0
        
        # 语义相关性加权
        if '？' in query or '?' in query:
            # 问句应该有明确回答
            if any(marker in answer for marker in ['是', '否', '有', '没有', '可以', '不能']):
                relevance += 0.2
        
        return min(1.0, relevance)
    
    def _assess_completeness(self, answer: str, query: str, context: str) -> float:
        """评估答案完整性"""
        if not answer:
            return 0.0
        
        # 基础长度评估
        length_score = min(1.0, len(answer) / 200.0)
        
        # 覆盖度评估：检查是否回答了查询的主要部分
        query_aspects = self._extract_query_aspects(query)
        covered_aspects = sum(1 for aspect in query_aspects 
                            if any(keyword in answer.lower() 
                                  for keyword in aspect.split()))
        coverage_score = covered_aspects / len(query_aspects) if query_aspects else 0.8
        
        # 结构化程度
        structure_score = 0.8
        if '：' in answer or ':' in answer:
            structure_score += 0.1
        if any(marker in answer for marker in ['首先', '其次', '最后', '另外']):
            structure_score += 0.1
        
        return (length_score * 0.3 + coverage_score * 0.5 + structure_score * 0.2)
    
    def _assess_accuracy(self, answer: str, sources: List[Dict]) -> float:
        """评估答案准确性"""
        if not sources or not answer:
            return 0.5  # 无法验证时给中等分数
        
        # 提取答案中的关键事实
        answer_facts = self._extract_facts(answer)
        verified_facts = 0
        total_facts = len(answer_facts)
        
        # 与源文档对比验证
        for fact in answer_facts:
            for source in sources:
                source_text = source.get('text', '').lower()
                if any(keyword in source_text for keyword in fact.lower().split()):
                    verified_facts += 1
                    break
        
        return verified_facts / total_facts if total_facts > 0 else 0.7
    
    def _assess_coherence(self, answer: str) -> float:
        """评估答案连贯性"""
        if not answer:
            return 0.0
        
        sentences = re.split(r'[.!?。！？]', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 0.8  # 单句默认连贯
        
        coherence_score = 0.8
        
        # 检查句子间的连贯性
        transitions = ['因此', '所以', '然而', '但是', '另外', '此外', '同时']
        has_transitions = any(trans in answer for trans in transitions)
        if has_transitions:
            coherence_score += 0.1
        
        # 检查重复和冗余
        unique_sentences = set(sentences)
        if len(unique_sentences) < len(sentences) * 0.8:
            coherence_score -= 0.2  # 重复过多
        
        return min(1.0, coherence_score)
    
    def _check_logical_consistency(self, answer: str, context: str) -> Dict[str, Any]:
        """逻辑一致性检查"""
        results = {
            'overall_score': 0.85,  # 提高默认分数，减少重试
            'issues': [],
            'checks_performed': []
        }
        
        # 执行各种一致性检查
        for check_name, check_func in self.consistency_checkers.items():
            try:
                check_result = check_func(answer, context)
                results['checks_performed'].append(check_name)
                
                if not check_result['is_consistent']:
                    results['issues'].append(f"{check_name}一致性问题: {check_result['description']}")
                    results['overall_score'] -= 0.2
            except Exception as e:
                print(f"⚠️ 一致性检查{check_name}失败: {str(e)}")
        
        results['overall_score'] = max(0.0, results['overall_score'])
        return results
    
    def _check_temporal_consistency(self, answer: str, context: str) -> Dict[str, Any]:
        """时间一致性检查"""
        # 提取时间信息
        time_patterns = [
            r'(20\d{2})年',
            r'(19\d{2})年', 
            r'(\d{1,2})月',
            r'(\d{1,2})日'
        ]
        
        answer_times = []
        context_times = []
        
        for pattern in time_patterns:
            answer_times.extend(re.findall(pattern, answer))
            context_times.extend(re.findall(pattern, context))
        
        # 简单的时间一致性检查
        if answer_times and context_times:
            inconsistent_times = set(answer_times) - set(context_times)
            if inconsistent_times:
                return {
                    'is_consistent': False,
                    'description': f'时间信息不一致: {inconsistent_times}'
                }
        
        return {'is_consistent': True, 'description': '时间信息一致'}
    
    def _check_numerical_consistency(self, answer: str, context: str) -> Dict[str, Any]:
        """数值一致性检查"""
        # 提取数值信息
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        answer_numbers = set(re.findall(number_pattern, answer))
        context_numbers = set(re.findall(number_pattern, context))
        
        if answer_numbers and context_numbers:
            # 检查是否有明显错误的数值
            inconsistent_numbers = answer_numbers - context_numbers
            if len(inconsistent_numbers) > len(answer_numbers) * 0.5:
                return {
                    'is_consistent': False,
                    'description': f'数值信息可能不准确: {inconsistent_numbers}'
                }
        
        return {'is_consistent': True, 'description': '数值信息基本一致'}
    
    def _check_categorical_consistency(self, answer: str, context: str) -> Dict[str, Any]:
        """分类一致性检查"""
        # 检查技术栈分类
        tech_categories = {
            'frontend': ['react', 'vue', 'angular', 'javascript', 'typescript'],
            'backend': ['python', 'java', 'nodejs', 'golang', 'rust'],
            'database': ['mysql', 'postgresql', 'mongodb', 'redis']
        }
        
        answer_lower = answer.lower()
        for category, terms in tech_categories.items():
            mentioned_terms = [term for term in terms if term in answer_lower]
            if len(mentioned_terms) > 1:
                # 如果提到多个同类技术，检查是否合理
                if category in ['frontend', 'backend'] and len(mentioned_terms) > 2:
                    return {
                        'is_consistent': False,
                        'description': f'{category}技术栈提及过多可能存在混淆'
                    }
        
        return {'is_consistent': True, 'description': '分类信息基本合理'}
    
    def _check_causal_consistency(self, answer: str, context: str) -> Dict[str, Any]:
        """因果关系一致性检查"""
        causal_patterns = [
            r'因为.*所以',
            r'由于.*导致',
            r'因此.*结果'
        ]
        
        causal_statements = []
        for pattern in causal_patterns:
            matches = re.findall(pattern, answer, re.DOTALL)
            causal_statements.extend(matches)
        
        # 这里可以添加更复杂的因果关系验证逻辑
        return {'is_consistent': True, 'description': '因果关系基本合理'}
    
    def _detect_quality_issues(self, answer: str) -> List[str]:
        """检测质量问题"""
        issues = []
        
        for issue_type, patterns in self.quality_issues_patterns.items():
            for pattern in patterns:
                if re.search(pattern, answer, re.IGNORECASE):
                    issues.append(f"{issue_type}: {pattern}匹配")
                    break
        
        return issues
    
    def _extract_query_aspects(self, query: str) -> List[str]:
        """提取查询要点"""
        # 简单的关键词提取
        aspects = []
        
        # 提取问题关键词
        question_words = ['什么', '如何', '为什么', '哪个', '多少']
        for word in question_words:
            if word in query:
                # 提取问题词后的内容作为一个方面
                parts = query.split(word)
                if len(parts) > 1:
                    aspects.append(word + parts[1].split('？')[0].split('?')[0])
        
        # 提取技术术语
        tech_terms = re.findall(r'\b[a-zA-Z]{3,}\b', query)
        aspects.extend(tech_terms)
        
        # 提取中文关键词
        chinese_terms = re.findall(r'[\u4e00-\u9fff]{2,}', query)
        aspects.extend(chinese_terms)
        
        return aspects[:5]  # 限制数量
    
    def _extract_facts(self, text: str) -> List[str]:
        """提取文本中的事实陈述"""
        facts = []
        
        # 提取包含数值的句子
        sentences = re.split(r'[.!?。！？]', text)
        for sentence in sentences:
            if re.search(r'\d+', sentence):
                facts.append(sentence.strip())
        
        # 提取明确的陈述句
        definitive_patterns = [
            r'.*是.*',
            r'.*使用.*',
            r'.*包含.*',
            r'.*支持.*'
        ]
        
        for sentence in sentences:
            for pattern in definitive_patterns:
                if re.search(pattern, sentence.strip()):
                    facts.append(sentence.strip())
                    break
        
        return facts[:10]  # 限制数量
    
    def _generate_improvement_suggestions(self, scores: Dict[str, float], issues: List[str]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        if scores.get('relevance', 0) < 0.5:
            suggestions.append('增强答案与问题的相关性')
        
        if scores.get('completeness', 0) < 0.5:
            suggestions.append('提供更完整和详细的回答')
        
        if scores.get('accuracy', 0) < 0.5:
            suggestions.append('验证答案中的事实准确性')
        
        if scores.get('coherence', 0) < 0.5:
            suggestions.append('改善答案的逻辑结构和连贯性')
        
        if 'incomplete_answer' in str(issues):
            suggestions.append('避免使用模糊或不确定的表达')
        
        if 'contradictory_info' in str(issues):
            suggestions.append('检查并解决信息冲突')
        
        return suggestions
    
    def auto_correction_attempt(self, 
                              original_answer: str,
                              quality_assessment: Dict[str, Any],
                              context: str,
                              query: str) -> Dict[str, Any]:
        """自动纠错尝试"""
        correction_result = {
            'corrected_answer': original_answer,
            'corrections_made': [],
            'improvement_achieved': False
        }
        
        # 基于质量评估结果进行针对性修正
        issues = quality_assessment.get('quality_issues', [])
        
        corrected_answer = original_answer
        
        # 1. 处理不完整回答
        if any('incomplete_answer' in issue for issue in issues):
            if len(original_answer) < 100:
                corrected_answer = self._expand_brief_answer(corrected_answer, context)
                correction_result['corrections_made'].append('扩展简短回答')
        
        # 2. 处理模糊回答
        if any('vague_response' in issue for issue in issues):
            corrected_answer = self._clarify_vague_statements(corrected_answer)
            correction_result['corrections_made'].append('澄清模糊表述')
        
        # 3. 处理逻辑不一致
        if quality_assessment.get('component_scores', {}).get('consistency', 1.0) < 0.6:
            corrected_answer = self._fix_logical_inconsistencies(corrected_answer)
            correction_result['corrections_made'].append('修正逻辑不一致')
        
        correction_result['corrected_answer'] = corrected_answer
        correction_result['improvement_achieved'] = len(correction_result['corrections_made']) > 0
        
        return correction_result
    
    def _expand_brief_answer(self, answer: str, context: str) -> str:
        """扩展简短回答"""
        if len(answer) < 50:
            # 从上下文中提取补充信息
            context_sentences = re.split(r'[.!?。！？]', context)
            relevant_sentences = context_sentences[:2]  # 取前两句作为补充
            
            if relevant_sentences:
                expanded = answer + " " + " ".join(relevant_sentences)
                return expanded
        return answer
    
    def _clarify_vague_statements(self, answer: str) -> str:
        """澄清模糊表述"""
        vague_replacements = {
            r'可能': '根据文档信息',
            r'大概': '大约',
            r'或许': '基于现有信息',
            r'似乎': '显示为'
        }
        
        clarified_answer = answer
        for vague, clear in vague_replacements.items():
            clarified_answer = re.sub(vague, clear, clarified_answer)
        
        return clarified_answer
    
    def _fix_logical_inconsistencies(self, answer: str) -> str:
        """修正逻辑不一致"""
        # 简单的逻辑修正：移除明显矛盾的表述
        contradictory_patterns = [
            r'但是.*然而',
            r'不是.*是'
        ]
        
        fixed_answer = answer
        for pattern in contradictory_patterns:
            if re.search(pattern, fixed_answer):
                # 简单处理：保留前半部分
                parts = re.split(pattern, fixed_answer)
                if len(parts) > 1:
                    fixed_answer = parts[0].strip() + "。"
        
        return fixed_answer

class OllamaClient:
    """Ollama LLM客户端 - 优化版，支持问题分类和专业Prompt"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model = "llama3.2:1b"  # 使用轻量级模型
        
        # 专业化Prompt模板
        self.prompt_templates = {
            'numerical_info': '''你是一个专业的数据分析助手。请仔细查找并提取上下文中的准确数值信息。

⚠️ 数值查询关键要求：
1. 精确匹配：优先寻找完全匹配的数值表达
2. 比较分析：如果涉及比较，需要列出所有相关数值进行对比
3. 数据验证：检查数值的一致性和准确性
4. 排序处理：如果问"最多"、"最少"等，需要明确排序

分析步骤：
1. 扫描每个文档片段中的所有数值（star数、fork数、commit数、时间等）
2. 提取项目名称和对应的数值信息
3. 如果是比较查询，列出所有项目的相关数值
4. 按照查询要求排序或筛选结果

上下文信息：
{context}

问题：{query}

请按以下格式回答：
**直接答案**: [直接回答问题]
**数值详情**: 
- 项目A: [具体数值]
- 项目B: [具体数值] 
- 项目C: [具体数值]
**数据来源**: [数据出处]
**比较结果**: [如果是比较查询，明确给出排序结果]''',

            'component_analysis': '''你是一个资深的软件架构师。请深入分析技术组件和系统架构。

分析维度：
- **核心组件**: 识别主要技术组件
- **架构模式**: 分析设计模式和架构风格
- **技术选型**: 评估技术栈的选择理由
- **实现细节**: 关注具体的实现方式

上下文信息：
{context}

问题：{query}

请按以下结构回答：
## 技术组件分析
### 主要组件
- [组件1]: [功能说明]
- [组件2]: [功能说明]

### 架构特点
- [特点1]: [详细说明]
- [特点2]: [详细说明]

### 实现细节
[具体的实现方式和技术细节]''',

            'language_tech': '''你是一个编程语言专家。请准确识别和分析项目中使用的编程语言和技术栈。

关注重点：
- 主要编程语言(前端/后端)
- 框架和库的使用
- 开发工具和构建系统
- 数据库和存储技术

上下文信息：
{context}

问题：{query}

请按以下格式精确回答：
**主要语言**: [按使用比重排序]
**前端技术**: [具体技术栈]
**后端技术**: [具体技术栈]
**数据存储**: [数据库等]
**其他工具**: [构建工具、部署等]''',

            'tech_comparison': '''你是一个技术咨询专家。请进行深入的技术对比分析。

对比维度：
- 功能特性差异
- 技术架构对比
- 适用场景分析
- 优缺点评估

上下文信息：
{context}

问题：{query}

请按以下结构回答：
## 技术对比分析
### 主要差异
- [差异1]: [详细说明]
- [差异2]: [详细说明]

### 适用场景
- [技术A]: [适合的场景]
- [技术B]: [适合的场景]

### 综合评估
[总结性的分析结论]''',

            'project_info': '''你是一个项目管理专家。请准确提取项目的基本信息。

关注要点：
- 项目基本属性(名称、描述、时间等)
- 开发状态和活跃度
- 社区影响力指标
- 项目规模信息

上下文信息：
{context}

问题：{query}

请简洁明确地回答问题，重点突出关键信息。''',

            'general': '''基于以下上下文信息，准确回答用户的问题。

上下文信息：
{context}

用户问题：{query}

请基于上述上下文信息简洁地回答问题。如果上下文中没有相关信息，请说明无法基于提供的信息回答。'''
        }
        
    def classify_question_advanced(self, query: str) -> str:
        """智能问题分类 - 增强版，更好地识别查询类型"""
        query_lower = query.lower()
        
        # 1. 数值查询识别 - 提高优先级
        numerical_patterns = [
            r'(?:多少|数量|几个|多大|多长|排行|最多|最少).*(?:star|fork|commit|项目)',
            r'(?:star|fork|commit).*(?:数|量|个数|多少|几个)',
            r'哪个.*(?:最多|最少|最大|最小|排名|第一)',
            r'\d+.*(?:star|fork|commit|个|次)',
            r'(?:总共|一共|总计).*(?:多少|几个|数量)',
            r'(?:排序|排名|排行榜)',
            r'(?:什么时候|何时|哪年|哪月).*(?:创建|发布|更新)',
            r'(?:创建|发布|更新).*(?:时间|日期)'
        ]
        
        for pattern in numerical_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                print(f"🔢 问题分类: numerical_info - 匹配模式: {pattern}")
                return 'numerical_info'
        
        # 2. 技术对比查询
        comparison_patterns = [
            r'区别|对比|差异|不同|优缺点|vs|versus|compare|比较',
            r'(?:哪个|哪些).*(?:更好|更适合|推荐)',
            r'异同|相同|不同点|相似'
        ]
        
        for pattern in comparison_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                print(f"🔍 问题分类: tech_comparison - 匹配模式: {pattern}")
                return 'tech_comparison'
        
        # 3. 组件架构分析
        component_patterns = [
            r'组件|模块|架构|技术栈|框架|实现方式|工作原理|系统设计',
            r'包含|使用.*(?:技术|框架|库)',
            r'如何.*(?:实现|工作|运行)'
        ]
        
        for pattern in component_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                print(f"🔍 问题分类: component_analysis - 匹配模式: {pattern}")
                return 'component_analysis'
        
        # 4. 编程语言技术查询
        language_patterns = [
            r'编程语言|开发语言|技术|框架|library|framework',
            r'用什么语言|用的是什么|采用.*语言',
            r'什么.*(?:技术栈|框架|语言)'
        ]
        
        for pattern in language_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                print(f"🔍 问题分类: language_tech - 匹配模式: {pattern}")
                return 'language_tech'
        
        # 5. 项目信息查询
        project_patterns = [
            r'项目|仓库|github|基本信息',
            r'介绍|简介|说明|描述',
            r'是什么|做什么|用途'
        ]
        
        for pattern in project_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                print(f"🔍 问题分类: project_info - 匹配模式: {pattern}")
                return 'project_info'
        
        print("🔍 问题分类: general - 未匹配特定模式")
        return 'general'
    
    def validate_answer_quality(self, answer: str, query: str, question_type: str) -> Dict[str, Any]:
        """答案质量验证机制"""
        validation_results = {
            'relevance_score': 0.0,
            'completeness_score': 0.0,
            'accuracy_indicators': [],
            'quality_issues': [],
            'confidence_level': 'medium'
        }
        
        # 基础相关性检查
        query_keywords = set(re.findall(r'\w+', query.lower()))
        answer_keywords = set(re.findall(r'\w+', answer.lower()))
        common_keywords = query_keywords & answer_keywords
        relevance = len(common_keywords) / len(query_keywords) if query_keywords else 0.0
        validation_results['relevance_score'] = relevance
        
        # 完整性检查(基于问题类型)
        if question_type == 'numerical_info':
            has_numbers = bool(re.search(r'\d+', answer))
            has_units = bool(re.search(r'star|fork|commit|个|年|月|日', answer, re.IGNORECASE))
            validation_results['completeness_score'] = 1.0 if has_numbers and has_units else 0.3
            if has_numbers:
                validation_results['accuracy_indicators'].append('包含数值信息')
            else:
                validation_results['quality_issues'].append('缺少数值信息')
                
        elif question_type == 'component_analysis':
            has_structure = bool(re.search(r'组件|模块|架构|系统', answer))
            has_details = len(answer) > 100
            validation_results['completeness_score'] = 0.8 if has_structure and has_details else 0.4
            
        elif question_type == 'language_tech':
            tech_terms = ['python', 'javascript', 'typescript', 'react', 'java', 'golang', 'rust']
            mentioned_techs = [term for term in tech_terms if term in answer.lower()]
            validation_results['completeness_score'] = min(len(mentioned_techs) / 3.0, 1.0)
            if mentioned_techs:
                validation_results['accuracy_indicators'].extend(mentioned_techs)
                
        else:
            # 通用完整性检查
            validation_results['completeness_score'] = min(len(answer) / 100.0, 1.0)
        
        # 质量问题检查
        if len(answer) < 20:
            validation_results['quality_issues'].append('答案过短')
        if '抱歉' in answer and '无法' in answer:
            validation_results['quality_issues'].append('无法提供有效答案')
        if answer.count('，') < 2 and len(answer) > 50:
            validation_results['quality_issues'].append('缺少结构化表达')
        
        # 置信度评估
        if validation_results['relevance_score'] > 0.8 and validation_results['completeness_score'] > 0.7:
            validation_results['confidence_level'] = 'high'
        elif validation_results['relevance_score'] < 0.3 or validation_results['completeness_score'] < 0.3:
            validation_results['confidence_level'] = 'low'
            
        return validation_results
        
    async def generate_answer(self, context: str, query: str) -> str:
        """基于上下文生成回答 - 使用智能Prompt选择和质量验证"""
        try:
            # 1. 问题分类
            question_type = self.classify_question_advanced(query)
            
            # 2. 选择合适的Prompt模板
            prompt_template = self.prompt_templates.get(question_type, self.prompt_templates['general'])
            prompt = prompt_template.format(context=context, query=query)
            
            print(f"🎯 使用Prompt类型: {question_type}")
            
            # 3. LLM生成
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.05,  # 优化：进一步降低温度加快生成
                        "top_p": 0.7,  # 优化：降低top_p加快生成,
                        "max_tokens": 400 if question_type in ['component_analysis', 'tech_comparison'] else 200  # 优化：减少token数
                    }
                }
                
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=20)  # 优化：平衡超时时间到20秒
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        answer = result.get("response", "抱歉，无法生成回答。")
                        
                        # 4. 答案质量验证
                        quality_metrics = self.validate_answer_quality(answer, query, question_type)
                        print(f"📊 答案质量评估: 相关性={quality_metrics['relevance_score']:.2f}, "
                              f"完整性={quality_metrics['completeness_score']:.2f}, "
                              f"置信度={quality_metrics['confidence_level']}")
                        
                        if quality_metrics['quality_issues']:
                            print(f"⚠️  质量问题: {', '.join(quality_metrics['quality_issues'])}")
                        
                        return answer
                    else:
                        error_text = await response.text()
                        print(f"LLM生成失败: {response.status} - {error_text}")
                        return "抱歉，LLM服务暂时不可用。"
                        
        except Exception as e:
            print(f"调用LLM服务失败: {str(e)}")
            return f"生成回答时出错：{str(e)}"

class CrossEncoderReranker:
    """
    Cross-Encoder重排序器 - Task3 RAG优化核心组件
    
    实现查询-文档对的精确相关性评分，提供二阶段重排序：
    1. 第一阶段：混合检索（向量+BM25）召回候选文档
    2. 第二阶段：Cross-Encoder对查询-文档对进行精确评分和重排序
    
    预期效果：准确率提升20%，同时保持<200ms的重排序延迟
    """
    
    def __init__(self, 
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 max_length: int = 512,
                 batch_size: int = 8,
                 cache_size: int = 1000):
        """
        初始化Cross-Encoder重排序器
        
        Args:
            model_name: 预训练模型名称，默认使用轻量级高效模型
            max_length: 文本最大长度限制
            batch_size: 批处理大小，优化推理性能
            cache_size: 缓存大小，避免重复计算
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.model = None
        self.enabled = CROSS_ENCODER_AVAILABLE
        
        # 性能监控
        self.rerank_count = 0
        self.total_rerank_time = 0.0
        self.cache_hits = 0
        
        # LRU缓存用于避免重复计算
        self._score_cache = {}
        self.cache_size = cache_size
        
        if self.enabled:
            self._initialize_model()
        else:
            print("⚠️  Cross-Encoder重排序器已禁用（依赖未安装）")
    
    def _initialize_model(self):
        """延迟初始化模型，避免启动时的阻塞"""
        try:
            print(f"🔄 初始化Cross-Encoder模型: {self.model_name}")
            self.model = CrossEncoder(self.model_name, max_length=self.max_length)
            print(f"✅ Cross-Encoder模型加载完成")
        except Exception as e:
            print(f"❌ Cross-Encoder模型加载失败: {e}")
            self.enabled = False
    
    def _generate_cache_key(self, query: str, doc_text: str) -> str:
        """生成缓存键"""
        combined = f"{query}||{doc_text[:200]}"  # 限制长度避免键过长
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def _clean_cache(self):
        """清理缓存，保持在指定大小内"""
        if len(self._score_cache) > self.cache_size:
            # 移除最旧的一半条目
            keys_to_remove = list(self._score_cache.keys())[:len(self._score_cache)//2]
            for key in keys_to_remove:
                del self._score_cache[key]
    
    def _preprocess_text(self, text: str) -> str:
        """文本预处理，优化中文支持"""
        if not text:
            return ""
        
        # 移除多余空白字符
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 截断过长文本，保留重要信息
        if len(text) > self.max_length - 100:  # 为查询留出空间
            text = text[:self.max_length - 100] + "..."
        
        return text
    
    async def rerank_documents(self, 
                              query: str, 
                              documents: List[Dict[str, Any]], 
                              top_k: int = 5) -> List[Dict[str, Any]]:
        """
        对检索到的文档进行重排序
        
        Args:
            query: 用户查询
            documents: 检索到的候选文档列表
            top_k: 返回的重排序后文档数量
            
        Returns:
            重排序后的文档列表，按相关性得分降序排列
        """
        if not self.enabled or not documents:
            return documents[:top_k]
        
        start_time = time.time()
        
        try:
            # 预处理查询和文档
            processed_query = self._preprocess_text(query)
            processed_docs = []
            cache_scores = {}
            pairs_to_score = []
            
            # 检查缓存并准备需要评分的对
            for i, doc in enumerate(documents):
                doc_text = self._preprocess_text(doc.get('text', ''))
                cache_key = self._generate_cache_key(processed_query, doc_text)
                
                if cache_key in self._score_cache:
                    cache_scores[i] = self._score_cache[cache_key]
                    self.cache_hits += 1
                else:
                    pairs_to_score.append((i, processed_query, doc_text))
                
                processed_docs.append(doc)
            
            # 批量计算未缓存的评分
            new_scores = {}
            if pairs_to_score:
                pairs_for_model = [(query_text, doc_text) for _, query_text, doc_text in pairs_to_score]
                
                # 分批处理以控制内存使用
                batch_scores = []
                for i in range(0, len(pairs_for_model), self.batch_size):
                    batch = pairs_for_model[i:i + self.batch_size]
                    batch_result = self.model.predict(batch)
                    batch_scores.extend(batch_result)
                
                # 存储新评分到缓存
                for (doc_idx, query_text, doc_text), score in zip(pairs_to_score, batch_scores):
                    new_scores[doc_idx] = float(score)
                    cache_key = self._generate_cache_key(query_text, doc_text)
                    self._score_cache[cache_key] = float(score)
            
            # 合并所有评分
            all_scores = {**cache_scores, **new_scores}
            
            # 为文档添加重排序评分
            scored_documents = []
            for i, doc in enumerate(processed_docs):
                doc_copy = doc.copy()
                doc_copy['rerank_score'] = all_scores.get(i, 0.0)
                # 保留原始检索评分作为备用
                doc_copy['original_score'] = doc.get('score', 0.0)
                scored_documents.append(doc_copy)
            
            # 按重排序评分降序排列
            reranked_docs = sorted(scored_documents, 
                                 key=lambda x: x['rerank_score'], 
                                 reverse=True)
            
            # 更新性能统计
            self.rerank_count += 1
            rerank_time = time.time() - start_time
            self.total_rerank_time += rerank_time
            
            # 清理缓存
            self._clean_cache()
            
            print(f"🎯 Cross-Encoder重排序完成: {len(documents)}→{min(top_k, len(reranked_docs))} "
                  f"耗时: {rerank_time:.3f}s, 缓存命中: {self.cache_hits}/{self.rerank_count}")
            
            return reranked_docs[:top_k]
            
        except Exception as e:
            print(f"❌ Cross-Encoder重排序失败: {e}")
            # 降级到原始排序
            return documents[:top_k]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        if self.rerank_count == 0:
            return {"enabled": self.enabled, "rerank_count": 0}
        
        avg_time = self.total_rerank_time / self.rerank_count
        cache_hit_rate = self.cache_hits / self.rerank_count if self.rerank_count > 0 else 0
        
        return {
            "enabled": self.enabled,
            "model_name": self.model_name,
            "rerank_count": self.rerank_count,
            "total_time": round(self.total_rerank_time, 3),
            "avg_time_per_request": round(avg_time, 3),
            "cache_hit_rate": round(cache_hit_rate, 3),
            "cache_size": len(self._score_cache)
        }

class BM25TextRetriever:
    """BM25文本检索器 - 优化版"""
    
    def __init__(self):
        self.corpus_documents = []  # 存储文档内容
        self.bm25_model = None
        self.is_initialized = False
        # 专业术语词典 - 提升编程语言和技术术语识别（包含数值相关术语）
        self.tech_terms = {
            'react', 'typescript', 'python', 'fastapi', 'nextjs', 'prisma', 'milvus', 'ollama',
            'javascript', 'java', 'golang', 'rust', 'vue', 'angular', 'django', 'flask',
            'docker', 'kubernetes', 'mongodb', 'postgresql', 'redis', 'elasticsearch',
            'api', 'rest', 'graphql', 'microservice', 'frontend', 'backend', 'fullstack'
        }
        
        # 数值相关术语词典 - 专门用于数值查询优化
        self.numerical_terms = {
            'star', 'stars', 'fork', 'forks', 'commit', 'commits',
            '⭐', '个star', '颗星', '个fork', '次提交', '个提交',
            'github', 'repository', 'repo', 'project', '项目',
            '数量', '个数', '多少', '几个', '总数', '总共',
            '最多', '最少', '最大', '最小', '排名', '第一',
            '创建时间', '更新时间', '发布时间', '版本号'
        }
        
        # 项目名称模式 - 保持项目名称完整性
        self.project_name_patterns = [
            r'dify_wechat_plugin',
            r'rag_search', 
            r'thesis_work_flow',
            r'bikeread'
        ]
        # 初始化jieba分词器
        self._init_jieba()
        
    def _init_jieba(self):
        """初始化jieba分词器，添加专业术语"""
        # 添加技术术语到jieba词典
        for term in self.tech_terms:
            jieba.add_word(term)
            jieba.add_word(term.upper())  # 添加大写版本
            jieba.add_word(term.capitalize())  # 添加首字母大写版本
        
        # 添加数值相关术语到jieba词典
        for term in self.numerical_terms:
            jieba.add_word(term)
            jieba.add_word(term.upper()) if term.isascii() else None
        
        # 添加项目名称模式，确保完整性
        for pattern in self.project_name_patterns:
            jieba.add_word(pattern, freq=20000)  # 高频词，优先保持完整
    
    def extract_numerical_info(self, text: str) -> Dict[str, Any]:
        """提取数值信息 - 增强版，解决star数量等识别问题"""
        patterns = {
            'stars': [
                r'(\d+)\s*(?:star|⭐|stars|个star|颗星)',
                r'(?:star|⭐|stars).*?(\d+)',
                r'(\d+)\s*⭐',
                r'star.*?数量.*?(\d+)',
                r'(\d+)\s*(?:个|颗)?(?:star|⭐|stars)'
            ],
            'forks': [
                r'(\d+)\s*(?:fork|forks|个fork)',
                r'(?:fork|forks).*?(\d+)',
                r'fork.*?数量.*?(\d+)',
                r'(\d+)\s*个?fork'
            ],
            'commits': [
                r'(\d+)\s*(?:commit|commits|次提交|个提交)',
                r'(?:commit|commits).*?(\d+)',
                r'提交.*?(\d+)',
                r'(\d+)\s*(?:次|个)?(?:commit|提交)'
            ],
            'years': [
                r'(20\d{2})年',
                r'(20\d{2})',
                r'年份.*?(20\d{2})',
                r'(?:创建|发布|更新).*?(20\d{2})'
            ],
            'versions': [
                r'v?(\d+\.\d+(?:\.\d+)?)',
                r'版本.*?(\d+\.\d+(?:\.\d+)?)',
                r'version.*?(\d+\.\d+(?:\.\d+)?)'
            ],
            'projects_count': [
                r'(\d+)\s*(?:个|项)?(?:项目|project)',
                r'(?:项目|project).*?(\d+)',
                r'总共.*?(\d+).*?项目'
            ],
            'numbers': [r'\b(\d+(?:\.\d+)?)\b']
        }
        
        extracted = {}
        for key, pattern_list in patterns.items():
            all_matches = []
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                all_matches.extend(matches)
            if all_matches:
                # 去重并保持顺序
                unique_matches = list(dict.fromkeys(all_matches))
                extracted[key] = unique_matches
                
        return extracted
    
    def _preprocess_text_v2(self, text: str) -> List[str]:
        """优化的文本预处理 - 支持中英文混合、专业术语和数值信息"""
        # 1. 提取和保护数值信息（增强版）
        numerical_info = self.extract_numerical_info(text)
        protected_numbers = []
        for key, values in numerical_info.items():
            for value in values:
                # 创建更多数值相关的token组合
                protected_numbers.extend([
                    f"{key}_{value}",  # 如: stars_41
                    value,             # 原始数值
                    f"{value}_{key}",  # 如: 41_stars
                    f"{value}{key}",   # 如: 41stars
                ])
        
        # 2. 保护专业术语（包括数值相关术语和项目名称）
        protected_terms = []
        text_lower = text.lower()
        
        # 优先保护项目名称（完整匹配）
        for pattern in self.project_name_patterns:
            if pattern.lower() in text_lower:
                protected_terms.append(pattern)
        
        # 技术术语
        for term in self.tech_terms:
            if term in text_lower:
                protected_terms.append(term)
        
        # 数值相关术语
        for term in self.numerical_terms:
            if term in text_lower:
                protected_terms.append(term)
        
        # 3. 特殊数值模式提取（增强数值查询支持）
        special_numerical_patterns = [
            r'(\d+)\s*(?:个|颗|次|项)?\s*(?:star|⭐|stars)',
            r'(\d+)\s*(?:个)?\s*(?:fork|forks)',  
            r'(\d+)\s*(?:次|个)?\s*(?:commit|commits|提交)',
            r'(20\d{2})\s*年',
            r'排名\s*(\d+)',
            r'第\s*(\d+)',
            r'最多\s*(\d+)',
            r'最少\s*(\d+)'
        ]
        
        for pattern in special_numerical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            protected_numbers.extend(matches)
        
        # 4. 中文分词处理
        chinese_tokens = list(jieba.cut(text, cut_all=False))
        
        # 5. 英文单词提取（保持原始大小写）
        english_tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]*\b', text)
        
        # 6. 数字提取
        numeric_tokens = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        
        # 7. 合并所有token并去重，过滤停用词
        all_tokens = chinese_tokens + english_tokens + numeric_tokens + protected_terms + protected_numbers
        
        # 8. 清理和过滤
        stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', 
                     'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        
        filtered_tokens = []
        for token in all_tokens:
            token_clean = token.strip().lower()
            if (len(token_clean) > 1 and 
                token_clean not in stop_words and 
                not re.match(r'^[\s\.,;:!?()\[\]{}"\'\/\\-]+$', token_clean)):
                filtered_tokens.append(token_clean)
        
        # 9. 去重但保持顺序
        seen = set()
        unique_tokens = []
        for token in filtered_tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)
        
        return unique_tokens
    
    def _preprocess_text(self, text: str) -> List[str]:
        """文本预处理入口 - 使用优化版本"""
        return self._preprocess_text_v2(text)
    
    def build_index(self, documents: List[Dict[str, Any]]):
        """构建BM25索引"""
        try:
            self.corpus_documents = documents
            
            # 提取文本并预处理
            tokenized_corpus = []
            for doc in documents:
                text = doc.get('text', '')
                tokens = self._preprocess_text(text)
                tokenized_corpus.append(tokens)
            
            if tokenized_corpus:
                self.bm25_model = BM25Okapi(tokenized_corpus)
                self.is_initialized = True
                print(f"✅ BM25索引构建完成，文档数量: {len(documents)}")
            else:
                print("⚠️  文档为空，无法构建BM25索引")
                
        except Exception as e:
            print(f"❌ 构建BM25索引失败: {str(e)}")
            self.is_initialized = False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """BM25搜索"""
        if not self.is_initialized or not self.bm25_model:
            return []
        
        try:
            # 预处理查询
            query_tokens = self._preprocess_text(query)
            if not query_tokens:
                return []
            
            # BM25搜索
            scores = self.bm25_model.get_scores(query_tokens)
            
            # 获取Top-K结果
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # 只返回有意义的匹配
                    doc = self.corpus_documents[idx].copy()
                    doc['bm25_score'] = float(scores[idx])
                    results.append(doc)
            
            return results
            
        except Exception as e:
            print(f"❌ BM25搜索失败: {str(e)}")
            return []

class HybridRetriever:
    """混合检索器 - 结合向量检索和BM25文本检索 (优化版)"""
    
    def __init__(self, vector_client: VectorServiceClient):
        self.vector_client = vector_client
        self.bm25_retriever = BM25TextRetriever()
        self.document_cache = []  # 缓存文档用于BM25索引
        
    async def initialize_bm25_index(self):
        """初始化BM25索引 - 从向量服务获取所有文档"""
        try:
            print("🔄 正在初始化BM25索引...")
            
            # 通过向量服务搜索获取现有文档
            # 使用一个通用查询来获取文档
            sample_results = await self.vector_client.search_vectors("文档", top_k=50)
            
            if sample_results:
                self.document_cache = sample_results
                self.bm25_retriever.build_index(sample_results)
                print(f"✅ BM25索引初始化完成，缓存文档: {len(sample_results)}")
            else:
                print("⚠️  未找到文档用于BM25索引初始化")
                
        except Exception as e:
            print(f"❌ BM25索引初始化失败: {str(e)}")
    
    async def hybrid_search(self, 
                           query: str, 
                           top_k: int = 5, 
                           vector_weight: float = 0.6,
                           bm25_weight: float = 0.4) -> List[Dict[str, Any]]:
        """
        混合检索：结合向量检索和BM25文本检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            vector_weight: 向量检索权重
            bm25_weight: BM25检索权重
        """
        try:
            print(f"🔍 开始混合检索: vector_weight={vector_weight}, bm25_weight={bm25_weight}")
            
            # 1. 向量检索
            print("📊 执行向量检索...")
            vector_results = await self.vector_client.search_vectors(query, top_k * 2)  # 获取更多结果用于融合
            
            # 2. BM25文本检索  
            print("📝 执行BM25检索...")
            bm25_results = self.bm25_retriever.search(query, top_k * 2)
            
            # 3. 结果融合 - 使用优化的RRF算法（支持动态权重）
            print("🔀 融合检索结果（优化版）...")
            fused_results = self._fuse_results_v2(
                vector_results, bm25_results, 
                query, top_k, use_dynamic_weights=True
            )
            
            print(f"✅ 混合检索完成，返回 {len(fused_results)} 个结果")
            return fused_results
            
        except Exception as e:
            print(f"❌ 混合检索失败: {str(e)}")
            # 降级到向量检索
            return await self.vector_client.search_vectors(query, top_k)
    
    def _is_numerical_query(self, query: str) -> bool:
        """判断是否为数值查询 - 增强版"""
        # 基础数值模式
        basic_numerical_patterns = [
            r'\d+.*(?:star|⭐|stars|fork|forks|commit|commits|年|版本|个|次)',
            r'(?:多少|几个|数量|个数|多大|多长|排行|最多|最少).*(?:star|fork|commit|项目|年|月|版本)',
            r'(?:star|fork|commit|项目).*(?:数|量|个数|多少|几个)',
            r'(?:最多|最少|排名|排行|第一|最大|最小).*(?:star|fork|commit)'
        ]
        
        # 比较和排序相关的数值查询
        comparative_patterns = [
            r'(?:哪个|哪些).*(?:最多|最少|最大|最小|排名|第一)',
            r'(?:比较|对比).*(?:数量|大小|多少)',
            r'(?:排序|排名|排行榜).*(?:star|fork|commit)',
            r'(?:top|前)\s*\d+',
            r'(?:总共|一共|总计).*(?:多少|几个|数量)'
        ]
        
        # 时间相关的数值查询
        temporal_patterns = [
            r'(?:什么时候|何时|哪年|哪月).*(?:创建|建立|发布|更新)',
            r'(?:创建|建立|发布|更新).*(?:时间|日期)',
            r'20\d{2}年.*(?:创建|发布|更新)',
            r'(?:最新|最早|最后).*(?:更新|发布|创建)'
        ]
        
        all_patterns = basic_numerical_patterns + comparative_patterns + temporal_patterns
        query_lower = query.lower()
        
        # 检查是否匹配任何数值查询模式
        is_numerical = any(re.search(pattern, query_lower, re.IGNORECASE) for pattern in all_patterns)
        
        # 额外检查：包含具体数字的查询
        contains_numbers = bool(re.search(r'\d+', query))
        
        # 包含比较词汇的查询
        contains_comparison = any(word in query_lower for word in 
                                ['最多', '最少', '最大', '最小', '排名', '第一', '哪个', '比较', '对比'])
        
        result = is_numerical or (contains_numbers and contains_comparison)
        
        if result:
            print(f"🔢 检测到数值查询: '{query}' - 匹配模式或包含数值+比较词汇")
        
        return result
    
    def _is_technical_query(self, query: str) -> bool:
        """判断是否为技术查询"""
        tech_keywords = ['编程语言', 'framework', 'library', 'api', '技术栈', 'architecture',
                        'react', 'python', 'javascript', 'typescript', 'java', 'golang',
                        '架构', '组件', '模块', '设计', '实现', '技术', '框架', '语言']
        query_lower = query.lower()
        return any(keyword.lower() in query_lower for keyword in tech_keywords)
    
    def _is_comparison_query(self, query: str) -> bool:
        """判断是否为对比查询"""
        comparison_keywords = [
            '区别', '差异', '不同', '对比', '比较', 'vs', 'versus', 'compare',
            '优缺点', '优势', '劣势', '异同', '相同', '不同点', '相似'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in comparison_keywords)
    
    def _get_dynamic_weights(self, query: str) -> tuple:
        """根据查询类型动态调整权重 - 优化版"""
        query_lower = query.lower()
        
        if self._is_numerical_query(query):
            # 数值查询进一步细分
            
            # 1. 精确数值比较查询 (如"哪个项目star最多")
            if any(word in query_lower for word in ['最多', '最少', '最大', '最小', '排名', '第一']):
                return 0.2, 0.8  # 更强调BM25精确匹配
            
            # 2. 包含具体数字的查询 (如"41个star")
            elif re.search(r'\d+', query):
                return 0.25, 0.75
            
            # 3. 一般数值查询 (如"多少个star")
            else:
                return 0.3, 0.7
                
        elif self._is_technical_query(query):
            # 技术查询也进行细分
            
            # 1. 架构分析类查询
            if any(word in query_lower for word in ['架构', '组件', '模块', '设计', '实现']):
                return 0.75, 0.25
            
            # 2. 编程语言和技术栈查询
            elif any(word in query_lower for word in ['语言', '技术栈', 'framework', 'library']):
                return 0.7, 0.3
            
            # 3. 一般技术查询
            else:
                return 0.8, 0.2
                
        elif self._is_comparison_query(query):
            # 对比查询需要平衡语义理解和精确匹配
            return 0.5, 0.5
            
        else:
            # 默认平衡权重
            return 0.6, 0.4
    
    def _fuse_results_v2(self, 
                        vector_results: List[Dict], 
                        bm25_results: List[Dict],
                        query: str,
                        top_k: int,
                        use_dynamic_weights: bool = True) -> List[Dict[str, Any]]:
        """优化的RRF算法 - 支持动态权重和参数调整"""
        # 动态权重调整
        if use_dynamic_weights:
            vector_weight, bm25_weight = self._get_dynamic_weights(query)
        else:
            vector_weight, bm25_weight = 0.6, 0.4
        
        # 动态RRF参数：根据查询类型调整k值
        if self._is_numerical_query(query):
            k = 30  # 数值查询使用更小的k值，更重视排名靠前的结果
        elif self._is_comparison_query(query):
            k = 35  # 对比查询稍微平衡
        else:
            k = 40  # 默认值
        
        print(f"🔄 RRF融合参数: k={k}, vector_weight={vector_weight:.1f}, bm25_weight={bm25_weight:.1f}")
        
        document_scores = defaultdict(lambda: {'doc': None, 'final_score': 0.0, 'components': {}})
        
        # 处理向量检索结果
        for rank, result in enumerate(vector_results):
            doc_id = result.get('id', f"vector_{rank}")
            rrf_score = vector_weight * (1.0 / (k + rank + 1))
            
            document_scores[doc_id]['doc'] = result
            document_scores[doc_id]['final_score'] += rrf_score
            document_scores[doc_id]['components']['vector_rank'] = rank + 1
            document_scores[doc_id]['components']['vector_score'] = result.get('score', 0.0)
            document_scores[doc_id]['components']['vector_rrf'] = rrf_score
        
        # 处理BM25检索结果
        for rank, result in enumerate(bm25_results):
            doc_id = result.get('id', f"bm25_{rank}")
            rrf_score = bm25_weight * (1.0 / (k + rank + 1))
            
            if document_scores[doc_id]['doc'] is None:
                document_scores[doc_id]['doc'] = result
            
            document_scores[doc_id]['final_score'] += rrf_score
            document_scores[doc_id]['components']['bm25_rank'] = rank + 1
            document_scores[doc_id]['components']['bm25_score'] = result.get('bm25_score', 0.0)
            document_scores[doc_id]['components']['bm25_rrf'] = rrf_score
        
        # 记录动态权重信息
        for doc_data in document_scores.values():
            doc_data['components']['dynamic_vector_weight'] = vector_weight
            doc_data['components']['dynamic_bm25_weight'] = bm25_weight
            doc_data['components']['query_type'] = (
                'numerical' if self._is_numerical_query(query) else
                'technical' if self._is_technical_query(query) else
                'general'
            )
        
        # 按最终分数排序
        sorted_docs = sorted(
            document_scores.values(),
            key=lambda x: x['final_score'],
            reverse=True
        )
        
        # 构建最终结果
        final_results = []
        for item in sorted_docs[:top_k]:
            doc = item['doc'].copy()
            doc['hybrid_score'] = item['final_score']
            doc['retrieval_components'] = item['components']
            final_results.append(doc)
        
        return final_results
    
    def _fuse_results(self, 
                     vector_results: List[Dict], 
                     bm25_results: List[Dict],
                     vector_weight: float,
                     bm25_weight: float, 
                     top_k: int) -> List[Dict[str, Any]]:
        """倒数排名融合(RRF)算法融合检索结果 - 保持向后兼容"""
        k = 40  # 优化的RRF参数
        document_scores = defaultdict(lambda: {'doc': None, 'final_score': 0.0, 'components': {}})
        
        # 处理向量检索结果
        for rank, result in enumerate(vector_results):
            doc_id = result.get('id', f"vector_{rank}")
            rrf_score = vector_weight * (1.0 / (k + rank + 1))
            
            document_scores[doc_id]['doc'] = result
            document_scores[doc_id]['final_score'] += rrf_score
            document_scores[doc_id]['components']['vector_rank'] = rank + 1
            document_scores[doc_id]['components']['vector_score'] = result.get('score', 0.0)
            document_scores[doc_id]['components']['vector_rrf'] = rrf_score
        
        # 处理BM25检索结果
        for rank, result in enumerate(bm25_results):
            doc_id = result.get('id', f"bm25_{rank}")
            rrf_score = bm25_weight * (1.0 / (k + rank + 1))
            
            if document_scores[doc_id]['doc'] is None:
                document_scores[doc_id]['doc'] = result
            
            document_scores[doc_id]['final_score'] += rrf_score
            document_scores[doc_id]['components']['bm25_rank'] = rank + 1
            document_scores[doc_id]['components']['bm25_score'] = result.get('bm25_score', 0.0)
            document_scores[doc_id]['components']['bm25_rrf'] = rrf_score
        
        # 按最终分数排序
        sorted_docs = sorted(
            document_scores.values(),
            key=lambda x: x['final_score'],
            reverse=True
        )
        
        # 构建最终结果
        final_results = []
        for item in sorted_docs[:top_k]:
            doc = item['doc'].copy()
            doc['hybrid_score'] = item['final_score']
            doc['retrieval_components'] = item['components']
            final_results.append(doc)
        
        return final_results

app = FastAPI(title="RAG Service", version="1.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局客户端实例
vector_client = VectorServiceClient()
llm_client = OllamaClient()
hybrid_retriever = HybridRetriever(vector_client)

# Task3 - Cross-Encoder重排序组件
cross_encoder_reranker = CrossEncoderReranker(
    model_name=os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    batch_size=int(os.getenv("CROSS_ENCODER_BATCH_SIZE", "8")),
    cache_size=int(os.getenv("CROSS_ENCODER_CACHE_SIZE", "1000"))
)

# Task4 - 多轮查询机制组件
query_optimizer = QueryOptimizer()
multiround_retriever = MultiRoundRetriever(hybrid_retriever, query_optimizer)
context_assessor = ContextQualityAssessor()

# Task5 - 质量控制系统组件
answer_quality_controller = AnswerQualityController()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    retrieval_mode: str = "hybrid"  # "hybrid", "vector", "bm25"
    vector_weight: float = 0.6  # 向量检索权重
    bm25_weight: float = 0.4    # BM25检索权重
    # Task3 - Cross-Encoder重排序选项
    enable_reranking: bool = True  # 启用Cross-Encoder重排序
    rerank_top_k: int = 10         # 重排序候选文档数量（通常是top_k的2-3倍）
    # Task4 - 多轮查询机制选项
    enable_multiround: bool = True  # 启用多轮查询
    max_rounds: int = 2             # 优化：最大查询轮数以3改为2
    quality_threshold: float = 0.6   # 查询质量阈值
    # Task5 - 质量控制系统选项
    enable_quality_control: bool = True  # 启用质量控制
    max_regeneration_attempts: int = 2   # 最大重新生成尝试次数
    quality_score_threshold: float = 0.5 # 优化：降低质量阈值，减少重试

class SourceDocument(BaseModel):
    id: str
    score: float
    text: str
    metadata: Dict[str, Any] = {}

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    query_time: float
    status: str
    metadata: Dict[str, Any] = {}
    # Task4 - 多轮查询结果信息
    multiround_info: Optional[Dict[str, Any]] = None  # 多轮查询详细信息
    context_quality: Optional[Dict[str, Any]] = None  # 上下文质量评估
    # Task5 - 质量控制系统结果信息
    quality_assessment: Optional[Dict[str, Any]] = None  # 答案质量评估
    regeneration_info: Optional[Dict[str, Any]] = None   # 重新生成信息

@app.on_event("startup")
async def startup_event():
    """服务启动时初始化BM25索引"""
    try:
        await hybrid_retriever.initialize_bm25_index()
        print("🚀 RAG Service启动完成，混合检索已就绪")
    except Exception as e:
        print(f"⚠️  启动时BM25索引初始化失败: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy", 
        "service": "rag-service",
        "version": "3.0.0",  # Task5版本升级
        "capabilities": [
            "混合检索查询 (向量+BM25)",
            "向量相似性搜索", 
            "BM25文本检索",
            "Cross-Encoder重排序",  # Task3新增
            "LLM答案生成",
            "RAG完整流程",
            "动态权重调整",
            "多轮查询机制",        # Task4
            "智能查询重写",        # Task4
            "上下文质量评估",       # Task4
            "答案质量控制",        # Task5新增
            "逻辑一致性检查",      # Task5新增
            "自动纠错机制",        # Task5新增
            "智能重新生成"         # Task5新增
        ],
        "retrieval_modes": ["hybrid", "vector", "bm25"],
        "bm25_initialized": hybrid_retriever.bm25_retriever.is_initialized,
        "multiround_features": {          # Task4
            "query_optimization": True,
            "context_assessment": True,
            "max_supported_rounds": 3
        },
        "quality_control_features": {     # Task5新增
            "answer_quality_assessment": True,
            "logical_consistency_check": True,
            "auto_correction": True,
            "regeneration_capability": True,
            "supported_quality_dimensions": [
                "relevance", "completeness", "accuracy", 
                "coherence", "consistency"
            ]
        },
        "cross_encoder_features": {        # Task3新增
            "enabled": cross_encoder_reranker.enabled,
            "model_name": cross_encoder_reranker.model_name if cross_encoder_reranker.enabled else None,
            "reranking_available": CROSS_ENCODER_AVAILABLE,
            "performance_stats": cross_encoder_reranker.get_performance_stats()
        }
    }

@app.get("/stats/cross-encoder")
async def get_cross_encoder_stats():
    """获取Cross-Encoder重排序器性能统计"""
    return {
        "service": "rag-service",
        "component": "cross-encoder-reranker",
        "timestamp": datetime.now().isoformat(),
        "stats": cross_encoder_reranker.get_performance_stats(),
        "dependencies_available": CROSS_ENCODER_AVAILABLE,
        "model_info": {
            "name": cross_encoder_reranker.model_name,
            "max_length": cross_encoder_reranker.max_length,
            "batch_size": cross_encoder_reranker.batch_size,
            "cache_size_limit": cross_encoder_reranker.cache_size
        } if cross_encoder_reranker.enabled else None
    }

def _build_improvement_prompt(context: str, query: str, suggestions: List[str]) -> str:
    """构建改进的Prompt用于重新生成"""
    improvement_instructions = "请特别注意以下改进要求：\n"
    for i, suggestion in enumerate(suggestions, 1):
        improvement_instructions += f"{i}. {suggestion}\n"
    
    improved_prompt = f"""
基于以下上下文信息，准确回答用户的问题。

{improvement_instructions}

上下文信息：
{context}

用户问题：{query}

请提供一个完整、准确、相关且结构清晰的回答。
"""
    return improved_prompt

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """执行RAG查询 - 支持Task4多轮查询机制"""
    start_time = time.time()
    
    try:
        query = request.query
        top_k = request.top_k
        retrieval_mode = request.retrieval_mode
        vector_weight = request.vector_weight
        bm25_weight = request.bm25_weight
        enable_multiround = request.enable_multiround
        max_rounds = min(request.max_rounds, 2)  # 优化：限制最大轮数为2
        
        print(f"🔍 开始RAG查询: '{query}' (top_k={top_k}, mode={retrieval_mode}, multiround={enable_multiround})")
        
        # Task4 - 多轮查询机制
        multiround_info = None
        if enable_multiround and retrieval_mode in ["hybrid", "vector"]:
            print("🔄 启用多轮查询机制")
            # 配置多轮检索器参数
            multiround_retriever.max_rounds = max_rounds
            
            # 执行多轮检索
            multiround_result = await multiround_retriever.multi_round_search(query, top_k)
            search_results = multiround_result['results']
            multiround_info = {
                'rounds_executed': multiround_result['rounds_executed'],
                'query_rounds': multiround_result['query_rounds'],
                'total_raw_results': multiround_result['total_raw_results'],
                'multiround_triggered': multiround_result['multiround_triggered']
            }
            
            print(f"✅ 多轮查询完成：执行{multiround_result['rounds_executed']}轮，"
                  f"融合{multiround_result['total_raw_results']}个结果为{len(search_results)}个")
        else:
            # 1. 传统单轮检索模式
            print(f"📋 第1阶段：{retrieval_mode}检索（单轮模式）")
            
            if retrieval_mode == "hybrid":
                search_results = await hybrid_retriever.hybrid_search(
                    query, top_k, vector_weight, bm25_weight
                )
            elif retrieval_mode == "vector":
                search_results = await vector_client.search_vectors(query, top_k)
            elif retrieval_mode == "bm25":
                search_results = hybrid_retriever.bm25_retriever.search(query, top_k)
            else:
                # 默认使用混合检索
                search_results = await hybrid_retriever.hybrid_search(
                    query, top_k, vector_weight, bm25_weight
                )
        
        if not search_results:
            print("⚠️  未找到相关文档")
            return QueryResponse(
                answer="抱歉，我无法在文档库中找到与您问题相关的信息。",
                sources=[],
                query_time=time.time() - start_time,
                status="no_results",
                metadata={
                    "search_results_count": 0,
                    "retrieval_mode": retrieval_mode,
                    "vector_weight": vector_weight,
                    "bm25_weight": bm25_weight,
                    "reranking_enabled": request.enable_reranking
                }
            )
        
        print(f"✅ 找到 {len(search_results)} 个相关文档片段")
        
        # Task3 - Cross-Encoder重排序
        rerank_info = {"enabled": False, "reranked": False}
        if request.enable_reranking and cross_encoder_reranker.enabled:
            print(f"🎯 启动Cross-Encoder重排序 (候选文档: {len(search_results)}, 目标: {top_k})")
            
            try:
                # 获取更多候选文档用于重排序，但不超过现有数量
                rerank_candidates = min(request.rerank_top_k, len(search_results))
                candidates_for_rerank = search_results[:rerank_candidates]
                
                # 执行重排序
                reranked_results = await cross_encoder_reranker.rerank_documents(
                    query=query,
                    documents=candidates_for_rerank,
                    top_k=top_k
                )
                
                # 更新检索结果
                search_results = reranked_results
                rerank_info = {
                    "enabled": True,
                    "reranked": True,
                    "candidates_count": rerank_candidates,
                    "final_count": len(reranked_results),
                    "performance": cross_encoder_reranker.get_performance_stats()
                }
                
                print(f"✅ Cross-Encoder重排序完成: {rerank_candidates}→{len(reranked_results)} 文档")
                
            except Exception as e:
                print(f"⚠️  Cross-Encoder重排序失败，使用原始排序: {e}")
                rerank_info = {
                    "enabled": True,
                    "reranked": False,
                    "error": str(e)
                }
        else:
            if not request.enable_reranking:
                print("📋 Cross-Encoder重排序已禁用")
            elif not cross_encoder_reranker.enabled:
                print("⚠️  Cross-Encoder重排序不可用（依赖未安装）")
        
        # 2. 构建上下文
        print("📝 第2阶段：构建上下文")
        contexts = []
        sources = []
        
        for i, result in enumerate(search_results):
            text = result.get("text", "").strip()
            if text:  # 只处理非空文本
                contexts.append(f"文档片段{i+1}：{text}")
                
                # 处理不同检索模式的分数
                score = result.get("hybrid_score", 
                        result.get("score", 
                        result.get("bm25_score", 0.0)))
                
                # 添加检索组件信息到metadata
                metadata = result.get("metadata", {}).copy()
                if "retrieval_components" in result:
                    metadata["retrieval_info"] = result["retrieval_components"]
                metadata["retrieval_mode"] = retrieval_mode
                
                sources.append(SourceDocument(
                    id=result.get("id", f"doc_{i}"),
                    score=score,
                    text=text,
                    metadata=metadata
                ))
        
        if not contexts:
            print("⚠️  检索到的文档片段为空")
            return QueryResponse(
                answer="检索到的文档片段内容为空，无法生成回答。",
                sources=sources,
                query_time=time.time() - start_time,
                status="empty_context",
                metadata={"search_results_count": len(search_results)}
            )
        
        context = "\n\n".join(contexts)
        print(f"📄 上下文构建完成，总长度: {len(context)} 字符")
        
        # Task4 - 上下文质量评估
        context_quality = None
        if enable_multiround:
            # 将SourceDocument对象转换为字典用于评估
            sources_dict = [{"metadata": src.metadata, "text": src.text} for src in sources]
            context_quality = context_assessor.assess_context_quality(context, query, sources_dict)
            print(f"📊 上下文质量评估: 长度={context_quality['length_score']:.2f}, "
                  f"相关性={context_quality['relevance_score']:.2f}, "
                  f"覆盖率={context_quality['coverage_score']:.2f}")
            
            if context_quality.get('enhancement_suggestions'):
                print(f"💡 改进建议: {', '.join(context_quality['enhancement_suggestions'])}")
        
        # 3. LLM生成阶段（含Task5质量控制）
        print("🤖 第3阶段：LLM答案生成")
        
        final_answer = await llm_client.generate_answer(context, query)
        quality_assessment = None
        regeneration_info = None
        
        # Task5 - 质量控制系统
        if request.enable_quality_control:
            print("🔍 Task5质量控制：开始答案质量评估")
            
            # 初始质量评估
            sources_dict = [{"text": src.text, "metadata": src.metadata} for src in sources]
            quality_assessment = answer_quality_controller.comprehensive_quality_assessment(
                final_answer, query, context, sources_dict
            )
            
            print(f"📊 质量评估结果: 综合分数={quality_assessment['overall_score']:.2f}, "
                  f"置信度={quality_assessment['confidence_level']}")
            
            regeneration_attempts = 0
            max_attempts = min(request.max_regeneration_attempts, 1)  # 优化：减少重试次数
            regeneration_history = []
            
            # 质量控制循环
            # 优化：只在质量真的很差时才重试
            while quality_assessment['requires_regeneration'] and \
                  regeneration_attempts < max_attempts:
                
                regeneration_attempts += 1
                print(f"🔄 质量不达标，进行第{regeneration_attempts}次改进尝试")
                
                # 记录当前尝试
                attempt_info = {
                    'attempt': regeneration_attempts,
                    'previous_score': quality_assessment['overall_score'],
                    'issues_found': quality_assessment['quality_issues'],
                    'improvement_method': 'regeneration'
                }
                
                # 首先尝试自动纠错
                correction_result = answer_quality_controller.auto_correction_attempt(
                    final_answer, quality_assessment, context, query
                )
                
                if correction_result['improvement_achieved']:
                    print(f"✅ 自动纠错成功: {correction_result['corrections_made']}")
                    final_answer = correction_result['corrected_answer']
                    attempt_info['improvement_method'] = 'auto_correction'
                    attempt_info['corrections_made'] = correction_result['corrections_made']
                else:
                    # 如果自动纠错无效，则重新生成
                    print("🔄 自动纠错无效，重新生成答案")
                    
                    # 构建改进的Prompt
                    improvement_prompt = _build_improvement_prompt(
                        context, query, quality_assessment['improvement_suggestions']
                    )
                    final_answer = await llm_client.generate_answer(improvement_prompt, query)
                    attempt_info['improvement_method'] = 'regeneration_with_improved_prompt'
                
                # 重新评估质量
                quality_assessment = answer_quality_controller.comprehensive_quality_assessment(
                    final_answer, query, context, sources_dict
                )
                
                attempt_info['new_score'] = quality_assessment['overall_score']
                attempt_info['score_improvement'] = attempt_info['new_score'] - attempt_info['previous_score']
                regeneration_history.append(attempt_info)
                
                print(f"📊 改进后质量评估: 分数={quality_assessment['overall_score']:.2f}, "
                      f"提升={attempt_info['score_improvement']:.2f}")
            
            # 构建重新生成信息
            if regeneration_attempts > 0:
                regeneration_info = {
                    'total_attempts': regeneration_attempts,
                    'final_score_improvement': (
                        quality_assessment['overall_score'] - regeneration_history[0]['previous_score']
                        if regeneration_history else 0
                    ),
                    'regeneration_history': regeneration_history,
                    'final_quality_achieved': quality_assessment['overall_score'] >= request.quality_score_threshold
                }
                print(f"✅ 质量控制完成: {regeneration_attempts}次尝试, "
                      f"最终分数={quality_assessment['overall_score']:.2f}")
        
        query_time = time.time() - start_time
        print(f"✅ RAG查询完成，耗时: {query_time:.2f}秒")
        
        return QueryResponse(
            answer=final_answer,
            sources=sources,
            query_time=query_time,
            status="completed",
            metadata={
                "search_results_count": len(search_results),
                "context_length": len(context),
                "sources_count": len(sources),
                "retrieval_mode": retrieval_mode,
                "vector_weight": vector_weight,
                "bm25_weight": bm25_weight,
                "hybrid_retrieval_enabled": retrieval_mode == "hybrid",
                "bm25_initialized": hybrid_retriever.bm25_retriever.is_initialized,
                # Task3 新增元数据 - Cross-Encoder重排序
                "reranking_enabled": request.enable_reranking,
                "reranking_info": rerank_info,
                "rerank_top_k_configured": request.rerank_top_k,
                # Task4 新增元数据
                "multiround_enabled": enable_multiround,
                "max_rounds_configured": max_rounds,
                # Task5 新增元数据
                "quality_control_enabled": request.enable_quality_control,
                "quality_threshold_configured": request.quality_score_threshold
            },
            # Task4 新增字段
            multiround_info=multiround_info,
            context_quality=context_quality,
            # Task5 新增字段
            quality_assessment=quality_assessment,
            regeneration_info=regeneration_info
        )
        
    except Exception as e:
        query_time = time.time() - start_time
        print(f"❌ RAG查询失败: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"RAG查询失败: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)