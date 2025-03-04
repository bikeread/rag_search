from typing import Type
from .base import QueryProcessor
from .qa import QAQueryProcessor
from .comparison import ComparisonQueryProcessor
from .list import ListQueryProcessor
from .recommendation import RecommendationQueryProcessor
from .summary import SummaryQueryProcessor
from .chat import ChatQueryProcessor
from .emergency import EmergencyQueryProcessor

class QueryProcessorFactory:
    """
    查询处理器工厂。
    """
    
    _processors = {
        "qa": QAQueryProcessor,
        "comparison": ComparisonQueryProcessor,
        "list": ListQueryProcessor,
        "recommendation": RecommendationQueryProcessor,
        "summary": SummaryQueryProcessor,
        "chat": ChatQueryProcessor,
        "emergency": EmergencyQueryProcessor
    }
    
    @classmethod
    def create_processor(cls, processor_type: str) -> QueryProcessor:
        """
        创建查询处理器。

        Args:
            processor_type (str): 处理器类型

        Returns:
            QueryProcessor: 查询处理器实例
        """
        processor_class = cls._processors.get(processor_type)
        if not processor_class:
            raise ValueError(f"Unknown processor type: {processor_type}")
            
        return processor_class()
        
    @classmethod
    def get_processor_for_query(cls, query: str) -> QueryProcessor:
        """
        根据查询内容选择合适的处理器。

        Args:
            query (str): 用户查询

        Returns:
            QueryProcessor: 查询处理器实例
        """
        # 按优先级尝试各个处理器
        for processor_type in ["comparison", "list", "recommendation", "summary", "chat", "qa"]:
            processor = cls.create_processor(processor_type)
            if processor.can_handle(query):
                return processor
                
        # 如果没有合适的处理器，返回应急处理器
        return cls.create_processor("emergency") 