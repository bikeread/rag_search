"""
Field mapping management for document processing system.
"""

from typing import Dict, List, Set, Optional, Any
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import os
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

class FieldType(Enum):
    """字段类型枚举"""
    TEXT = auto()
    NUMERIC = auto()
    CATEGORY = auto()
    DATETIME = auto()
    BOOLEAN = auto()
    
    @classmethod
    def guess_type(cls, value: str) -> 'FieldType':
        """根据值推测字段类型
        
        Args:
            value: 字段值
            
        Returns:
            推测的字段类型
        """
        try:
            float(value)
            return cls.NUMERIC
        except ValueError:
            pass
            
        import re
        # 日期时间格式检查
        datetime_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
            r'\d{2}:\d{2}:\d{2}',  # HH:MM:SS
        ]
        if any(re.match(pattern, value) for pattern in datetime_patterns):
            return cls.DATETIME
            
        # 布尔值检查
        boolean_values = {'true', 'false', 'yes', 'no', '1', '0', 'y', 'n'}
        if value.lower() in boolean_values:
            return cls.BOOLEAN
            
        return cls.TEXT

@dataclass
class FieldValidation:
    """字段验证规则"""
    min_value: float = float('-inf')
    max_value: float = float('inf')
    max_length: int = 1000
    allow_decimal: bool = True
    date_format: str = "YYYY-MM-DD"

@dataclass
class FieldInfo:
    """字段信息类"""
    name: str  # 标准字段名
    type: FieldType  # 字段类型
    aliases: Set[str] = field(default_factory=set)  # 字段别名集合
    description: str = ""  # 字段描述
    sample_values: Set[str] = field(default_factory=set)  # 样本值集合
    statistics: Dict[str, float] = field(default_factory=dict)  # 统计信息
    validation: FieldValidation = field(default_factory=FieldValidation)
    category_mappings: Dict[str, List[str]] = field(default_factory=dict)

class FieldMappingManager:
    """字段映射管理器"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls, config_path: Optional[str] = None) -> 'FieldMappingManager':
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls(config_path)
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化字段映射管理器
        
        Args:
            config_path: 可选的配置文件路径，用于加载预定义的字段映射
        """
        self._field_mappings: Dict[str, FieldInfo] = {}
        self._field_groups: Dict[str, Set[str]] = defaultdict(set)  # 字段分组
        self._value_to_field_cache: Dict[str, str] = {}  # 值到字段的缓存
        
        # 加载默认配置
        default_config = Path(__file__).parent.parent.parent / 'config' / 'field_mappings.json'
        if config_path:
            self._load_config(config_path)
        elif default_config.exists():
            self._load_config(str(default_config))
    
    def _load_config(self, config_path: str):
        """从配置文件加载字段映射
        
        Args:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # 加载字段类型和验证规则
            for type_name, type_config in config.get('field_types', {}).items():
                field_type = FieldType[type_name.upper()]
                validation = FieldValidation(**type_config.get('validation', {}))
                
                for field_name in type_config.get('fields', []):
                    self.register_field(FieldInfo(
                        name=field_name,
                        type=field_type,
                        validation=validation
                    ))
            
            # 加载字段统计信息
            for field_name, stats in config.get('field_statistics', {}).items():
                if field_name in self._field_mappings:
                    self._field_mappings[field_name].statistics.update(stats)
            
            # 加载类别映射
            for category, aliases in config.get('category_mappings', {}).items():
                for field_info in self._field_mappings.values():
                    if field_info.type == FieldType.CATEGORY:
                        field_info.category_mappings[category] = aliases
            
            # 加载字段别名
            for field_name, aliases in config.get('field_aliases', {}).items():
                if field_name in self._field_mappings:
                    self._field_mappings[field_name].aliases.update(aliases)
            
            logger.info(f"Loaded field mappings from {config_path}")
            
        except Exception as e:
            logger.error(f"Error loading field mappings from {config_path}: {str(e)}")
    
    def save_config(self, config_path: str):
        """保存字段映射到配置文件
        
        Args:
            config_path: 配置文件路径
        """
        try:
            config = {
                'version': '1.0',
                'field_types': defaultdict(lambda: {'fields': [], 'validation': {}}),
                'field_statistics': {},
                'category_mappings': {},
                'field_aliases': {}
            }
            
            # 组织字段类型和验证规则
            for field_name, info in self._field_mappings.items():
                type_name = info.type.name.lower()
                config['field_types'][type_name]['fields'].append(field_name)
                config['field_types'][type_name]['validation'] = {
                    'min': info.validation.min_value,
                    'max': info.validation.max_value,
                    'max_length': info.validation.max_length,
                    'allow_decimal': info.validation.allow_decimal,
                    'date_format': info.validation.date_format
                }
            
            # 添加统计信息
            for field_name, info in self._field_mappings.items():
                if info.statistics:
                    config['field_statistics'][field_name] = info.statistics
            
            # 添加类别映射
            for field_name, info in self._field_mappings.items():
                if info.category_mappings:
                    config['category_mappings'].update(info.category_mappings)
            
            # 添加字段别名
            for field_name, info in self._field_mappings.items():
                if info.aliases:
                    config['field_aliases'][field_name] = list(info.aliases)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
                
            logger.info(f"Saved field mappings to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving field mappings to {config_path}: {str(e)}")
    
    def learn_from_data(self, data: List[Dict[str, str]], sample_size: int = 1000):
        """从数据中学习字段映射
        
        Args:
            data: 数据列表，每个数据项是字段名到值的映射
            sample_size: 用于分析的样本大小
        """
        try:
            # 收集字段信息
            field_samples: Dict[str, List[str]] = defaultdict(list)
            
            # 收集样本
            for item in data[:sample_size]:
                for field, value in item.items():
                    if isinstance(value, str):
                        field_samples[field].append(value.strip())
            
            # 分析每个字段
            for field, samples in field_samples.items():
                if not samples:
                    continue
                    
                # 统计最常见的类型
                type_counts = defaultdict(int)
                for value in samples:
                    field_type = FieldType.guess_type(value)
                    type_counts[field_type] += 1
                
                # 选择最常见的类型
                field_type = max(type_counts.items(), key=lambda x: x[1])[0]
                
                # 创建或更新字段信息
                if field not in self._field_mappings:
                    self.register_field(FieldInfo(
                        name=field,
                        type=field_type,
                        sample_values=set(samples[:10])  # 保存一些样本值
                    ))
                else:
                    # 更新现有字段信息
                    field_info = self._field_mappings[field]
                    field_info.type = field_type
                    field_info.sample_values.update(samples[:10])
                
                # 更新统计信息
                if field_type == FieldType.NUMERIC:
                    try:
                        numeric_values = [float(v) for v in samples if v]
                        if numeric_values:
                            stats = {
                                'min': min(numeric_values),
                                'max': max(numeric_values),
                                'avg': sum(numeric_values) / len(numeric_values)
                            }
                            self._field_mappings[field].statistics.update(stats)
                    except ValueError:
                        pass
            
            # 尝试识别相关字段并分组
            self._identify_field_groups()
            
            logger.info(f"Learned field mappings from {len(data)} records")
            
        except Exception as e:
            logger.error(f"Error learning field mappings: {str(e)}")
    
    def _identify_field_groups(self):
        """识别相关字段并分组"""
        # 基于字段名相似性分组
        name_groups = defaultdict(set)
        for field in self._field_mappings:
            # 提取字段名的主要部分（例如：product_name -> product）
            base_name = field.split('_')[0]
            name_groups[base_name].add(field)
        
        # 基于字段类型分组
        type_groups = defaultdict(set)
        for field, info in self._field_mappings.items():
            type_groups[info.type.name.lower()].add(field)
        
        # 更新字段分组
        self._field_groups.update(name_groups)
        self._field_groups.update(type_groups)
    
    def register_field(self, field_info: FieldInfo):
        """注册新的字段信息
        
        Args:
            field_info: 字段信息对象
        """
        self._field_mappings[field_info.name] = field_info
        logger.debug(f"Registered field: {field_info.name} with type: {field_info.type.name}")
    
    def get_standard_field_name(self, field: str) -> str:
        """获取标准字段名
        
        Args:
            field: 输入的字段名或别名
            
        Returns:
            标准字段名
        """
        field = field.lower().strip()
        
        # 检查是否是标准字段名
        if field in self._field_mappings:
            return field
            
        # 在所有字段的别名中查找
        for std_name, field_info in self._field_mappings.items():
            if field in field_info.aliases:
                return std_name
                
        return field  # 如果找不到映射，返回原始字段名
    
    def get_field_info(self, field: str) -> Optional[FieldInfo]:
        """获取字段信息
        
        Args:
            field: 字段名或别名
            
        Returns:
            字段信息对象
        """
        std_name = self.get_standard_field_name(field)
        return self._field_mappings.get(std_name)
    
    def get_fields_by_type(self, field_type: FieldType) -> List[str]:
        """获取指定类型的所有字段
        
        Args:
            field_type: 字段类型
            
        Returns:
            指定类型的字段名列表
        """
        return [
            name for name, info in self._field_mappings.items()
            if info.type == field_type
        ]
    
    def get_fields_in_group(self, group: str) -> Set[str]:
        """获取指定分组中的所有字段
        
        Args:
            group: 分组名称
            
        Returns:
            分组中的字段集合
        """
        return self._field_groups.get(group, set())
    
    def suggest_field_type(self, field: str, sample_values: List[str]) -> FieldType:
        """根据样本值建议字段类型
        
        Args:
            field: 字段名
            sample_values: 样本值列表
            
        Returns:
            建议的字段类型
        """
        type_counts = defaultdict(int)
        for value in sample_values:
            field_type = FieldType.guess_type(value)
            type_counts[field_type] += 1
        
        return max(type_counts.items(), key=lambda x: x[1])[0]
    
    def get_field_statistics(self, field: str) -> Dict[str, float]:
        """获取字段的统计信息
        
        Args:
            field: 字段名
            
        Returns:
            统计信息字典
        """
        field_info = self.get_field_info(field)
        return field_info.statistics if field_info else {}

# 创建全局实例（不再预定义字段映射）
field_mapping_manager = FieldMappingManager()

# 为产品相关字段添加更多别名映射
product_id_field = FieldInfo(
    name="product_id", 
    type=FieldType.TEXT, 
    aliases={"id", "商品ID", "编号", "产品序列号", "序列号", "产品唯一标识符"}
)
field_mapping_manager.register_field(product_id_field)

product_name_field = FieldInfo(
    name="product_name", 
    type=FieldType.TEXT, 
    aliases={"name", "商品名称", "名称", "产品", "产品名称"}
)
field_mapping_manager.register_field(product_name_field)

category_field = FieldInfo(
    name="category", 
    type=FieldType.CATEGORY, 
    aliases={"分类", "种类", "商品类别", "产品类别", "产品类型", "类型"}
)
field_mapping_manager.register_field(category_field)

price_field = FieldInfo(
    name="price", 
    type=FieldType.NUMERIC, 
    aliases={"售价", "单价", "金额", "产品价格", "价格"}
)
field_mapping_manager.register_field(price_field)

stock_field = FieldInfo(
    name="stock", 
    type=FieldType.NUMERIC, 
    aliases={"库存量", "存货", "产品库存", "库存"}
)
field_mapping_manager.register_field(stock_field)

description_field = FieldInfo(
    name="description", 
    type=FieldType.TEXT, 
    aliases={"商品描述", "详情", "介绍", "产品描述", "描述"}
)
field_mapping_manager.register_field(description_field)

logger.info("扩展字段映射别名配置完成") 