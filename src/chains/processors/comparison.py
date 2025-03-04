from typing import Any, Dict, List, Optional, Tuple
from .base import QueryProcessor
from src.utils.logging_utils import log_args_and_result  # 导入日志装饰器

class ComparisonQueryProcessor(QueryProcessor):
    """
    比较查询处理器。
    用于处理涉及比较的查询，如价格比较、性能比较等。
    """
    
    @log_args_and_result
    def process(
        self,
        query: str,
        documents: List[Any],
        chat_history: Optional[List[Tuple[str, str]]] = None,
        chain_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理比较查询。

        Args:
            query (str): 用户查询
            documents (List[Any]): 相关文档列表
            chat_history (Optional[List[Tuple[str, str]]], optional): 聊天历史. 默认为 None.
            chain_type (Optional[str], optional): 链类型. 默认为 None.
            **kwargs: 额外参数

        Returns:
            Dict[str, Any]: 处理结果
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if not documents:
            logger.warning("没有找到相关文档")
            return {
                "answer": "抱歉，没有找到可以比较的信息。",
                "sources": []
            }
        
        logger.info(f"处理查询: '{query}'，找到 {len(documents)} 个文档")
            
        # 处理最贵/最便宜的笔记本电脑查询

            
        # 其他比较查询的处理
        logger.warning(f"不支持的比较查询类型: {query}")
        return {
            "answer": "抱歉，目前不支持这种类型的比较查询。",
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
        # 检查查询是否包含比较关键词
        return any(keyword in query for keyword in "")