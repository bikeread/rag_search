from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

class QueryProcessor(ABC):
    """
    查询处理器基类。
    """
    
    @abstractmethod
    def process(
        self, 
        query: str, 
        documents: List[Any], 
        chat_history: Optional[List[Tuple[str, str]]] = None,
        chain_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理查询。

        Args:
            query (str): 用户查询
            documents (List[Any]): 相关文档列表
            chat_history (Optional[List[Tuple[str, str]]], optional): 聊天历史. 默认为 None.
            chain_type (Optional[str], optional): 链类型. 默认为 None.
            **kwargs: 额外参数

        Returns:
            Dict[str, Any]: 处理结果
        """
        pass

    @abstractmethod
    def can_handle(self, query: str) -> bool:
        """
        判断是否可以处理该查询。

        Args:
            query (str): 用户查询

        Returns:
            bool: 是否可以处理
        """
        pass 