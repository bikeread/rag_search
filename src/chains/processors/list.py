from typing import Any, Dict, List, Optional, Tuple
from .base import QueryProcessor

class ListQueryProcessor(QueryProcessor):
    """
    列表查询处理器。
    用于处理需要返回列表形式答案的查询，如推荐清单等。
    """
    
    def process(
        self,
        query: str,
        documents: List[Any],
        chat_history: Optional[List[Tuple[str, str]]] = None,
        chain_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理列表查询。

        Args:
            query (str): 用户查询
            documents (List[Any]): 相关文档列表
            chat_history (Optional[List[Tuple[str, str]]], optional): 聊天历史. 默认为 None.
            chain_type (Optional[str], optional): 链类型. 默认为 None.
            **kwargs: 额外参数

        Returns:
            Dict[str, Any]: 处理结果
        """
        if not documents:
            return {
                "answer": "抱歉，没有找到相关信息。",
                "sources": []
            }

        # 处理游戏本推荐查询
        if "游戏本" in query:
            gaming_laptops = []
            for doc in documents:
                try:
                    content = doc.page_content
                    metadata = doc.metadata
                    
                    # 从内容中提取信息
                    name_line = [line for line in content.split('\n') if '产品名称:' in line or '产品名称：' in line]
                    price_line = [line for line in content.split('\n') if '价格:' in line or '价格：' in line]
                    processor_line = [line for line in content.split('\n') if '处理器:' in line or '处理器：' in line]
                    memory_line = [line for line in content.split('\n') if '内存:' in line or '内存：' in line]
                    storage_line = [line for line in content.split('\n') if '硬盘:' in line or '硬盘：' in line]
                    graphics_line = [line for line in content.split('\n') if '显卡:' in line or '显卡：' in line]
                    features_line = [line for line in content.split('\n') if '特点:' in line or '特点：' in line]
                    
                    # 判断是否为游戏本
                    is_gaming = False
                    if features_line:
                        features = features_line[0].split(':')[-1].strip().lower()
                        if '游戏' in features or 'gaming' in features:
                            is_gaming = True
                    if graphics_line:
                        graphics = graphics_line[0].split(':')[-1].strip().lower()
                        if 'rtx' in graphics or 'gtx' in graphics:
                            is_gaming = True
                            
                    if is_gaming:
                        laptop_info = {
                            "name": name_line[0].split(':')[-1].strip() if name_line else "未知型号",
                            "price": price_line[0].split(':')[-1].strip() if price_line else "未知",
                            "processor": processor_line[0].split(':')[-1].strip() if processor_line else "",
                            "memory": memory_line[0].split(':')[-1].strip() if memory_line else "",
                            "storage": storage_line[0].split(':')[-1].strip() if storage_line else "",
                            "graphics": graphics_line[0].split(':')[-1].strip() if graphics_line else "",
                            "features": features_line[0].split(':')[-1].strip() if features_line else ""
                        }
                        gaming_laptops.append(laptop_info)
                except Exception as e:
                    continue

            if not gaming_laptops:
                return {
                    "answer": "抱歉，没有找到游戏本的信息。",
                    "sources": []
                }

            # 生成推荐列表
            answer = "以下是推荐的游戏本：\n\n"
            for i, laptop in enumerate(gaming_laptops, 1):
                answer += f"{i}. {laptop['name']}\n"
                answer += f"   价格：{laptop['price']}\n"
                answer += f"   配置：\n"
                if laptop['processor']:
                    answer += f"   - 处理器：{laptop['processor']}\n"
                if laptop['memory']:
                    answer += f"   - 内存：{laptop['memory']}\n"
                if laptop['storage']:
                    answer += f"   - 存储：{laptop['storage']}\n"
                if laptop['graphics']:
                    answer += f"   - 显卡：{laptop['graphics']}\n"
                if laptop['features']:
                    answer += f"   特点：{laptop['features']}\n"
                answer += "\n"

            return {
                "answer": answer.strip(),
                "sources": documents
            }

        # 处理其他列表查询
        return {
            "answer": "抱歉，目前不支持这种类型的列表查询。",
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
        # 检查查询是否包含列表相关关键词
        list_keywords = ["推荐", "有哪些", "列表", "清单"]
        return any(keyword in query for keyword in list_keywords) 