from typing import Any, Dict, List
from .base import QueryProcessor

class ChatQueryProcessor(QueryProcessor):
    """
    聊天查询处理器。
    用于处理对话式查询。
    """
    
    def process(self, query: str, documents: List[Any], **kwargs) -> Dict[str, Any]:
        """
        处理聊天查询。

        Args:
            query (str): 用户查询
            documents (List[Any]): 相关文档列表
            **kwargs: 额外参数

        Returns:
            Dict[str, Any]: 处理结果
        """
        chat_history = kwargs.get("chat_history", [])
        
        if not documents and not chat_history:
            return {
                "answer": "抱歉，我不太理解您的问题。能否换个方式描述？",
                "sources": []
            }
            
        # 这里应该实现实际的对话生成逻辑
        return {
            "answer": "根据对话历史和文档生成回答",
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
        # 检查是否是对话式查询
        chat_keywords = ["你", "您", "我", "聊天", "对话"]
        return any(keyword in query for keyword in chat_keywords) 