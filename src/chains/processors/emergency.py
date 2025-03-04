from typing import Any, Dict, List
from .base import QueryProcessor

class EmergencyQueryProcessor(QueryProcessor):
    """
    应急查询处理器。
    用于在其他处理器失败时提供基本的问答功能。
    """
    
    def process(self, query: str, documents: List[Any], **kwargs) -> Dict[str, Any]:
        """
        处理应急查询。

        Args:
            query (str): 用户查询
            documents (List[Any]): 相关文档列表
            **kwargs: 额外参数

        Returns:
            Dict[str, Any]: 处理结果
        """
        return {
            "answer": "抱歉，系统暂时无法处理您的请求。请稍后重试。",
            "sources": []
        }

    def can_handle(self, query: str) -> bool:
        """
        判断是否可以处理该查询。

        Args:
            query (str): 用户查询

        Returns:
            bool: 是否可以处理
        """
        # 应急处理器可以处理所有查询
        return True 