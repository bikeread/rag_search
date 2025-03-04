from typing import Any, Dict, List
from .base import QueryProcessor

class RecommendationQueryProcessor(QueryProcessor):
    """
    推荐查询处理器。
    用于处理需要推荐或建议的查询。
    """
    
    def process(self, query: str, documents: List[Any], **kwargs) -> Dict[str, Any]:
        """
        处理推荐查询。

        Args:
            query (str): 用户查询
            documents (List[Any]): 相关文档列表
            **kwargs: 额外参数

        Returns:
            Dict[str, Any]: 处理结果
        """
        if not documents:
            return {
                "answer": "抱歉，没有找到可以推荐的信息。",
                "sources": []
            }
            
        # 这里应该实现实际的推荐逻辑
        return {
            "answer": "根据文档生成推荐答案",
            "sources": documents
        }

    def can_handle(self, query: str) -> bool:
        """
        判断是否可以处理该查询。

        Args:
            query (str): 用户查询

        Returns:
            bool: 是否可以处理
        """
        # 检查查询是否包含推荐相关关键词
        recommendation_keywords = ["推荐", "建议", "介绍", "选择", "合适", "适合"]
        return any(keyword in query for keyword in recommendation_keywords) 