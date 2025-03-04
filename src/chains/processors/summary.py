from typing import Any, Dict, List
from .base import QueryProcessor

class SummaryQueryProcessor(QueryProcessor):
    """
    摘要查询处理器。
    用于处理需要总结或概述的查询。
    """
    
    def process(self, query: str, documents: List[Any], **kwargs) -> Dict[str, Any]:
        """
        处理摘要查询。

        Args:
            query (str): 用户查询
            documents (List[Any]): 相关文档列表
            **kwargs: 额外参数

        Returns:
            Dict[str, Any]: 处理结果
        """
        if not documents:
            return {
                "answer": "抱歉，没有找到可以总结的信息。",
                "sources": []
            }
            
        # 这里应该实现实际的摘要生成逻辑
        return {
            "answer": "根据文档生成摘要答案",
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
        # 检查查询是否包含摘要相关关键词
        summary_keywords = ["总结", "概述", "摘要", "简述", "归纳", "概括"]
        return any(keyword in query for keyword in summary_keywords) 