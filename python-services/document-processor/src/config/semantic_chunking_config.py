"""
语义分块配置文件
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import os

@dataclass
class SemanticChunkingConfig:
    """语义分块配置"""
    
    # 基础配置
    enable_semantic_chunking: bool = True
    vector_service_url: str = "http://localhost:8002"
    
    # 分块大小配置
    min_chunk_size: int = 200
    max_chunk_size: int = 1500
    target_chunk_size: int = 1000
    
    # 语义连贯性配置
    coherence_threshold: float = 0.7
    similarity_window_size: int = 3
    
    # 关键信息保护配置
    critical_info_protection: bool = True
    number_protection_enabled: bool = True
    entity_protection_enabled: bool = True
    formula_protection_enabled: bool = True
    
    # 重叠策略配置
    adaptive_overlap: bool = True
    min_overlap_size: int = 50
    max_overlap_size: int = 300
    
    # 质量评估配置
    enable_quality_assessment: bool = True
    quality_threshold: float = 0.7
    detailed_assessment: bool = False
    
    # 降级策略配置
    fallback_to_smart_chunking: bool = True
    fallback_chunk_size: int = 1000
    fallback_overlap: int = 200
    
    # 性能配置
    async_processing: bool = True
    max_concurrent_chunks: int = 10
    timeout_seconds: int = 30
    
    # 中文文本优化配置
    chinese_text_optimization: bool = True
    chinese_sentence_patterns: list = None
    mixed_language_support: bool = True
    
    def __post_init__(self):
        """初始化后处理"""
        if self.chinese_sentence_patterns is None:
            self.chinese_sentence_patterns = [
                r'[.!?。！？]\s+',
                r'[.!?。！？](?=\n)',
                r'[.!?。！？]$',
                r'；\s*',
                r'：\s*(?=\n)',
            ]
        
        # 从环境变量读取配置
        self.vector_service_url = os.getenv(
            'VECTOR_SERVICE_URL', self.vector_service_url
        )
        
        # 验证配置
        self._validate_config()
    
    def _validate_config(self):
        """验证配置参数"""
        if self.min_chunk_size >= self.max_chunk_size:
            raise ValueError("min_chunk_size must be less than max_chunk_size")
        
        if not (0.0 <= self.coherence_threshold <= 1.0):
            raise ValueError("coherence_threshold must be between 0.0 and 1.0")
        
        if self.min_overlap_size >= self.max_overlap_size:
            raise ValueError("min_overlap_size must be less than max_overlap_size")
        
        if not (0.0 <= self.quality_threshold <= 1.0):
            raise ValueError("quality_threshold must be between 0.0 and 1.0")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SemanticChunkingConfig':
        """从字典创建配置"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def update_from_env(self):
        """从环境变量更新配置"""
        env_mappings = {
            'SEMANTIC_CHUNKING_ENABLED': ('enable_semantic_chunking', bool),
            'VECTOR_SERVICE_URL': ('vector_service_url', str),
            'MIN_CHUNK_SIZE': ('min_chunk_size', int),
            'MAX_CHUNK_SIZE': ('max_chunk_size', int),
            'COHERENCE_THRESHOLD': ('coherence_threshold', float),
            'CRITICAL_INFO_PROTECTION': ('critical_info_protection', bool),
            'QUALITY_THRESHOLD': ('quality_threshold', float),
        }
        
        for env_var, (attr_name, attr_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    if attr_type == bool:
                        value = env_value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        value = attr_type(env_value)
                    setattr(self, attr_name, value)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid value for {env_var}: {env_value}")

# 预定义配置模板
CONFIG_TEMPLATES = {
    'default': SemanticChunkingConfig(),
    
    'high_quality': SemanticChunkingConfig(
        coherence_threshold=0.8,
        critical_info_protection=True,
        enable_quality_assessment=True,
        detailed_assessment=True,
        min_chunk_size=300,
        max_chunk_size=1200
    ),
    
    'fast_processing': SemanticChunkingConfig(
        enable_semantic_chunking=False,
        fallback_to_smart_chunking=True,
        enable_quality_assessment=False,
        async_processing=True,
        max_concurrent_chunks=20
    ),
    
    'chinese_optimized': SemanticChunkingConfig(
        chinese_text_optimization=True,
        mixed_language_support=True,
        coherence_threshold=0.65,  # 中文语义连贯性要求略低
        critical_info_protection=True,
        target_chunk_size=800  # 中文分块适当减小
    ),
    
    'technical_documents': SemanticChunkingConfig(
        critical_info_protection=True,
        number_protection_enabled=True,
        entity_protection_enabled=True,
        formula_protection_enabled=True,
        coherence_threshold=0.75,
        min_chunk_size=250,
        max_chunk_size=1800
    ),
    
    'memory_optimized': SemanticChunkingConfig(
        enable_semantic_chunking=True,
        async_processing=False,
        max_concurrent_chunks=5,
        detailed_assessment=False,
        min_chunk_size=150,
        max_chunk_size=1000
    )
}

def get_config(template_name: str = 'default', 
               custom_overrides: Optional[Dict[str, Any]] = None) -> SemanticChunkingConfig:
    """
    获取配置实例
    
    Args:
        template_name: 配置模板名称
        custom_overrides: 自定义覆盖参数
        
    Returns:
        配置实例
    """
    if template_name not in CONFIG_TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(CONFIG_TEMPLATES.keys())}")
    
    config = CONFIG_TEMPLATES[template_name]
    
    # 应用自定义覆盖
    if custom_overrides:
        config_dict = config.to_dict()
        config_dict.update(custom_overrides)
        config = SemanticChunkingConfig.from_dict(config_dict)
    
    # 从环境变量更新
    config.update_from_env()
    
    return config

# 导出默认配置
default_config = get_config('default')