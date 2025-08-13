"""
数值信息优化器
专门处理中文数字信息的检测、保护和优化分块
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class NumberType(Enum):
    """数字类型枚举"""
    INTEGER = "integer"          # 整数
    DECIMAL = "decimal"          # 小数
    PERCENTAGE = "percentage"    # 百分比
    CURRENCY = "currency"        # 货币
    DATE = "date"               # 日期
    TIME = "time"               # 时间
    STAR_COUNT = "star_count"   # GitHub star数量
    MEASUREMENT = "measurement"  # 测量值(带单位)
    RATIO = "ratio"             # 比率
    COUNT = "count"             # 计数

@dataclass
class NumericalEntity:
    """数值实体"""
    text: str                   # 原始文本
    value: Optional[float]      # 数值
    number_type: NumberType     # 数字类型
    start_pos: int             # 开始位置
    end_pos: int               # 结束位置
    context_before: str        # 前文上下文
    context_after: str         # 后文上下文
    unit: Optional[str] = None # 单位
    confidence: float = 1.0    # 置信度

@dataclass
class NumericalCluster:
    """数值聚类"""
    entities: List[NumericalEntity]
    cluster_type: str
    importance_score: float
    should_keep_together: bool

class ChineseNumericalOptimizer:
    """中文数值信息优化器"""
    
    def __init__(self, context_window: int = 50):
        self.context_window = context_window
        
        # 中文数字模式（优化版）
        self.chinese_number_patterns = {
            # 基础数字模式
            'basic_numbers': [
                r'\d+\.?\d*',                           # 基础数字
                r'[一二三四五六七八九十百千万亿兆]+',      # 中文数字
                r'\d+[,，]\d+(?:[,，]\d+)*',            # 千分位数字
            ],
            
            # 百分比和比率
            'percentages': [
                r'\d+\.?\d*%',                          # 百分比
                r'\d+\.?\d*％',                         # 中文百分号
                r'百分之\d+\.?\d*',                     # 百分之X
                r'\d+\.?\d*个百分点',                   # 百分点
                r'\d+\.?\d*‰',                          # 千分比
                r'\d+\.?\d*比\d+\.?\d*',                # X比Y
                r'\d+:\d+',                             # 比率
            ],
            
            # 货币数值
            'currency': [
                r'¥\d+\.?\d*[万亿千百十]?',             # 人民币
                r'\$\d+\.?\d*[万亿千百十KMGT]?',        # 美元
                r'€\d+\.?\d*[万亿千百十KMGT]?',         # 欧元
                r'\d+\.?\d*[万亿千百十]*元',            # X元
                r'\d+\.?\d*[万亿千百十]*美元',          # X美元
                r'\d+\.?\d*[万亿千百十]*人民币',        # X人民币
            ],
            
            # 计量单位
            'measurements': [
                r'\d+\.?\d*[万亿千百十]*[个只条件份台辆架艘]', # 数量单位
                r'\d+\.?\d*[万亿千百十]*[米厘毫微纳]米?',    # 长度单位
                r'\d+\.?\d*[万亿千百十]*[克千克吨磅盎司]',   # 重量单位
                r'\d+\.?\d*[万亿千百十]*[升毫升加仑]',      # 体积单位
                r'\d+\.?\d*[万亿千百十]*[秒分钟小时天年]',  # 时间单位
                r'\d+\.?\d*[万亿千百十]*[字节BKMGT]B?',     # 存储单位
                r'\d+\.?\d*[万亿千百十]*[瓦特千瓦兆瓦]W?',  # 功率单位
            ],
            
            # 日期时间
            'datetime': [
                r'\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?',  # 日期
                r'\d{1,2}:\d{2}(?::\d{2})?',               # 时间
                r'\d{4}年\d{1,2}月\d{1,2}[日号]',          # 中文日期
                r'\d{1,2}月\d{1,2}[日号]',                 # 月日
                r'Q[1-4]\s*\d{4}',                         # 季度
                r'\d{4}Q[1-4]',                            # 年份季度
            ],
            
            # GitHub和技术指标
            'tech_metrics': [
                r'\d+\.?\d*[万千百十]*[⭐★stars?]',         # GitHub stars
                r'\d+\.?\d*[万千百十]*[forks?]',           # GitHub forks
                r'\d+\.?\d*[万千百十]*[commits?]',         # Git commits
                r'\d+\.?\d*[万千百十]*[contributors?]',    # 贡献者
                r'\d+\.?\d*[万千百十]*[issues?]',          # Issues
                r'\d+\.?\d*[万千百十]*[downloads?]',       # 下载量
                r'\d+\.?\d*[万千百十]*用户',               # 用户数
                r'\d+\.?\d*[万千百十]*次',                 # 次数
            ],
            
            # 排名和序号
            'rankings': [
                r'第\d+[名位]',                            # 第X名
                r'排名第\d+',                              # 排名第X
                r'NO\.\s*\d+',                             # NO.X
                r'#\d+',                                   # #X
                r'Top\s*\d+',                              # TopX
                r'前\d+[名位]',                            # 前X名
            ]
        }
        
        # 关键词指示符（帮助确定数字重要性）
        self.importance_indicators = {
            'high_importance': [
                '最多', '最少', '最高', '最低', '第一', '首位', '榜首',
                '冠军', '领先', '超过', '达到', '突破', '创新高', '记录'
            ],
            'comparison': [
                '比较', '对比', '相比', '超过', '低于', '高于', '等于',
                '增长', '下降', '提升', '减少', '翻倍', '折半'
            ],
            'statistical': [
                '平均', '总计', '累计', '统计', '分析', '数据', '指标',
                '比例', '占比', '份额', '概率', '频率'
            ]
        }
        
        # 编译正则表达式以提高性能
        self.compiled_patterns = {}
        for category, patterns in self.chinese_number_patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def extract_numerical_entities(self, text: str) -> List[NumericalEntity]:
        """提取数值实体"""
        entities = []
        
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entity = self._create_numerical_entity(
                        text, match, category
                    )
                    if entity:
                        entities.append(entity)
        
        # 去重和排序
        entities = self._deduplicate_entities(entities)
        entities.sort(key=lambda x: x.start_pos)
        
        logger.info(f"提取到 {len(entities)} 个数值实体")
        return entities
    
    def _create_numerical_entity(self, 
                                text: str, 
                                match: re.Match, 
                                category: str) -> Optional[NumericalEntity]:
        """创建数值实体"""
        start_pos = match.start()
        end_pos = match.end()
        entity_text = match.group()
        
        # 提取上下文
        context_start = max(0, start_pos - self.context_window)
        context_end = min(len(text), end_pos + self.context_window)
        
        context_before = text[context_start:start_pos]
        context_after = text[end_pos:context_end]
        
        # 确定数字类型
        number_type = self._classify_number_type(entity_text, category, context_before, context_after)
        
        # 提取数值
        numeric_value = self._extract_numeric_value(entity_text)
        
        # 提取单位
        unit = self._extract_unit(entity_text)
        
        # 计算重要性分数
        confidence = self._calculate_importance_score(
            entity_text, context_before, context_after
        )
        
        return NumericalEntity(
            text=entity_text,
            value=numeric_value,
            number_type=number_type,
            start_pos=start_pos,
            end_pos=end_pos,
            context_before=context_before,
            context_after=context_after,
            unit=unit,
            confidence=confidence
        )
    
    def _classify_number_type(self, 
                             entity_text: str, 
                             category: str, 
                             context_before: str, 
                             context_after: str) -> NumberType:
        """分类数字类型"""
        # 基于模式类别的初步分类
        category_mapping = {
            'percentages': NumberType.PERCENTAGE,
            'currency': NumberType.CURRENCY,
            'measurements': NumberType.MEASUREMENT,
            'datetime': NumberType.DATE,
            'tech_metrics': NumberType.COUNT,
            'rankings': NumberType.COUNT
        }
        
        if category in category_mapping:
            return category_mapping[category]
        
        # 基于文本特征的细分类
        if '%' in entity_text or '％' in entity_text or '百分' in entity_text:
            return NumberType.PERCENTAGE
        
        if any(symbol in entity_text for symbol in ['¥', '$', '€', '元', '美元']):
            return NumberType.CURRENCY
        
        if ':' in entity_text and len(entity_text.split(':')) == 2:
            return NumberType.TIME
        
        if re.search(r'\d{4}[-/年]', entity_text):
            return NumberType.DATE
        
        if '⭐' in entity_text or 'star' in entity_text.lower():
            return NumberType.STAR_COUNT
        
        if '.' in entity_text:
            return NumberType.DECIMAL
        
        return NumberType.INTEGER
    
    def _extract_numeric_value(self, entity_text: str) -> Optional[float]:
        """提取数值"""
        try:
            # 移除非数字字符，保留数字和小数点
            numeric_part = re.sub(r'[^\d.]', '', entity_text)
            
            if not numeric_part:
                return None
            
            # 处理中文数字
            if re.search(r'[一二三四五六七八九十百千万亿]', entity_text):
                return self._convert_chinese_number(entity_text)
            
            # 处理普通数字
            if '.' in numeric_part:
                return float(numeric_part)
            else:
                return float(numeric_part)
        
        except (ValueError, TypeError):
            return None
    
    def _convert_chinese_number(self, chinese_text: str) -> Optional[float]:
        """转换中文数字为阿拉伯数字"""
        # 简化的中文数字转换
        chinese_digits = {
            '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
            '五': 5, '六': 6, '七': 7, '八': 8, '九': 9
        }
        
        units = {
            '十': 10, '百': 100, '千': 1000, 
            '万': 10000, '亿': 100000000
        }
        
        # 这里只是一个简化实现
        # 实际应用中可能需要更复杂的中文数字解析
        try:
            # 如果包含阿拉伯数字，提取阿拉伯数字部分
            arabic_match = re.search(r'\d+', chinese_text)
            if arabic_match:
                base_value = float(arabic_match.group())
                
                # 检查单位
                for unit, multiplier in units.items():
                    if unit in chinese_text:
                        base_value *= multiplier
                        break
                
                return base_value
            
            return None
        
        except:
            return None
    
    def _extract_unit(self, entity_text: str) -> Optional[str]:
        """提取单位"""
        # 常见单位模式
        unit_patterns = [
            r'[%％]',                           # 百分比
            r'[¥$€]',                          # 货币符号
            r'[元美元人民币]',                  # 货币单位
            r'[万亿千百十]',                    # 中文数量单位
            r'[⭐★]',                           # 星星
            r'[个只条件份台辆架艘]',            # 数量单位
            r'[米厘毫微纳克升瓦特]',            # 度量单位
            r'[BKMGT]B?',                       # 存储单位
            r'[年月日时分秒]',                  # 时间单位
        ]
        
        for pattern in unit_patterns:
            match = re.search(pattern, entity_text)
            if match:
                return match.group()
        
        return None
    
    def _calculate_importance_score(self, 
                                  entity_text: str, 
                                  context_before: str, 
                                  context_after: str) -> float:
        """计算重要性分数"""
        score = 0.5  # 基础分数
        
        # 基于关键词指示符的分数
        full_context = context_before + entity_text + context_after
        
        for category, keywords in self.importance_indicators.items():
            for keyword in keywords:
                if keyword in full_context:
                    if category == 'high_importance':
                        score += 0.3
                    elif category == 'comparison':
                        score += 0.2
                    elif category == 'statistical':
                        score += 0.1
        
        # 基于数字特征的分数
        if '%' in entity_text or '百分' in entity_text:
            score += 0.2  # 百分比通常重要
        
        if '⭐' in entity_text or 'star' in entity_text.lower():
            score += 0.2  # GitHub stars重要
        
        if any(symbol in entity_text for symbol in ['¥', '$', '€', '万', '亿']):
            score += 0.15  # 大额货币重要
        
        # 基于位置的分数
        if '第一' in full_context or 'NO.1' in full_context or '#1' in full_context:
            score += 0.25  # 排名第一特别重要
        
        return min(score, 1.0)
    
    def _deduplicate_entities(self, entities: List[NumericalEntity]) -> List[NumericalEntity]:
        """去重数值实体"""
        if not entities:
            return entities
        
        # 按位置排序
        entities.sort(key=lambda x: (x.start_pos, x.end_pos))
        
        deduplicated = []
        
        for entity in entities:
            # 检查是否与已有实体重叠
            overlap_found = False
            
            for existing in deduplicated:
                # 检查位置重叠
                if (entity.start_pos < existing.end_pos and 
                    entity.end_pos > existing.start_pos):
                    
                    # 如果重叠，保留置信度更高的
                    if entity.confidence > existing.confidence:
                        deduplicated.remove(existing)
                        deduplicated.append(entity)
                    
                    overlap_found = True
                    break
            
            if not overlap_found:
                deduplicated.append(entity)
        
        return deduplicated
    
    def cluster_numerical_entities(self, entities: List[NumericalEntity]) -> List[NumericalCluster]:
        """聚类数值实体"""
        if not entities:
            return []
        
        clusters = []
        
        # 按类型分组
        type_groups = {}
        for entity in entities:
            if entity.number_type not in type_groups:
                type_groups[entity.number_type] = []
            type_groups[entity.number_type].append(entity)
        
        # 为每个类型创建聚类
        for number_type, group_entities in type_groups.items():
            if len(group_entities) >= 2:
                # 检查是否应该保持在一起（基于位置和上下文）
                should_keep_together = self._should_cluster_together(group_entities)
                
                # 计算聚类重要性
                importance_score = np.mean([e.confidence for e in group_entities])
                
                cluster = NumericalCluster(
                    entities=group_entities,
                    cluster_type=number_type.value,
                    importance_score=importance_score,
                    should_keep_together=should_keep_together
                )
                clusters.append(cluster)
        
        return clusters
    
    def _should_cluster_together(self, entities: List[NumericalEntity]) -> bool:
        """判断实体是否应该聚在一起"""
        if len(entities) < 2:
            return False
        
        # 计算实体间的平均距离
        positions = [e.start_pos for e in entities]
        positions.sort()
        
        total_distance = 0
        for i in range(1, len(positions)):
            total_distance += positions[i] - positions[i-1]
        
        avg_distance = total_distance / (len(positions) - 1)
        
        # 如果平均距离小于500个字符，认为应该聚在一起
        return avg_distance < 500
    
    def optimize_chunk_boundaries(self, 
                                 text: str, 
                                 initial_boundaries: List[int],
                                 entities: List[NumericalEntity]) -> List[int]:
        """优化分块边界以保护数值信息"""
        if not entities:
            return initial_boundaries
        
        optimized_boundaries = initial_boundaries.copy()
        
        # 检查每个边界是否分割了重要的数值信息
        for i, boundary in enumerate(initial_boundaries[1:-1], 1):  # 跳过首尾边界
            
            # 检查边界附近是否有数值实体
            nearby_entities = [
                e for e in entities 
                if abs(e.start_pos - boundary) < 100 or abs(e.end_pos - boundary) < 100
            ]
            
            if nearby_entities:
                # 找到最安全的新边界位置
                new_boundary = self._find_safe_boundary(
                    text, boundary, nearby_entities
                )
                
                if new_boundary != boundary:
                    optimized_boundaries[i] = new_boundary
                    logger.info(f"调整边界从 {boundary} 到 {new_boundary} 以保护数值信息")
        
        return optimized_boundaries
    
    def _find_safe_boundary(self, 
                           text: str, 
                           original_boundary: int, 
                           nearby_entities: List[NumericalEntity]) -> int:
        """找到安全的边界位置"""
        # 寻找最近的句子边界，避开数值实体
        search_range = 200  # 在原边界前后200字符内搜索
        
        start_search = max(0, original_boundary - search_range)
        end_search = min(len(text), original_boundary + search_range)
        
        # 句子边界模式
        sentence_boundaries = []
        
        sentence_patterns = [
            r'[.!?。！？]\s+',
            r'[.!?。！？](?=\n)',
            r'\n\s*\n',
            r'[；;]\s+'
        ]
        
        for pattern in sentence_patterns:
            for match in re.finditer(pattern, text[start_search:end_search]):
                boundary_pos = start_search + match.end()
                
                # 检查这个边界是否安全（不分割数值实体）
                is_safe = True
                for entity in nearby_entities:
                    if entity.start_pos <= boundary_pos <= entity.end_pos:
                        is_safe = False
                        break
                
                if is_safe:
                    sentence_boundaries.append(boundary_pos)
        
        # 选择最接近原边界的安全位置
        if sentence_boundaries:
            return min(sentence_boundaries, 
                      key=lambda x: abs(x - original_boundary))
        
        # 如果找不到安全的句子边界，返回原边界
        return original_boundary
    
    def generate_numerical_metadata(self, 
                                   chunk_text: str, 
                                   entities: List[NumericalEntity]) -> Dict[str, Any]:
        """生成数值相关的元数据"""
        # 过滤属于当前分块的实体
        chunk_entities = [
            e for e in entities
            if e.text in chunk_text
        ]
        
        if not chunk_entities:
            return {
                'contains_numbers': False,
                'numerical_density': 0.0,
                'number_types': [],
                'high_importance_numbers': 0
            }
        
        # 统计不同类型的数字
        type_counts = {}
        high_importance_count = 0
        
        for entity in chunk_entities:
            if entity.number_type.value not in type_counts:
                type_counts[entity.number_type.value] = 0
            type_counts[entity.number_type.value] += 1
            
            if entity.confidence > 0.7:
                high_importance_count += 1
        
        # 计算数值密度
        numerical_density = len(chunk_entities) / len(chunk_text) * 1000
        
        return {
            'contains_numbers': True,
            'numerical_density': numerical_density,
            'number_types': list(type_counts.keys()),
            'number_type_counts': type_counts,
            'high_importance_numbers': high_importance_count,
            'total_numbers': len(chunk_entities),
            'avg_number_confidence': np.mean([e.confidence for e in chunk_entities]),
            'numerical_entities': [
                {
                    'text': e.text,
                    'type': e.number_type.value,
                    'confidence': e.confidence,
                    'value': e.value,
                    'unit': e.unit
                }
                for e in chunk_entities
            ]
        }