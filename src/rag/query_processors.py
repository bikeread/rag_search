"""
Query processing strategies for different types of queries.
"""

from typing import List, Dict, Any, Optional, Type, Set
from abc import ABC, abstractmethod
import re
import logging
from langchain.docstore.document import Document
import jieba
from src.rag.field_mapping import field_mapping_manager, FieldType

logger = logging.getLogger(__name__)

class DocumentParser:
    """文档解析器，用于统一处理文档解析逻辑"""
    
    @staticmethod
    def parse_document(doc: Document) -> Dict[str, str]:
        """解析文档内容为字段映射
        
        Args:
            doc: 待解析的文档
            
        Returns:
            解析后的字段映射
        """
        doc_values = {}
        if " | " in doc.page_content:
            for item in doc.page_content.split(" | "):
                if ": " in item:
                    key, value = item.split(": ", 1)
                    # 使用字段映射管理器获取标准字段名
                    std_field = field_mapping_manager.get_standard_field_name(key.strip())
                    doc_values[std_field] = value.strip()
        return doc_values
    
    @staticmethod
    def extract_field_values(docs: List[Document], field: str) -> List[tuple[Document, float]]:
        """从文档中提取指定字段的值
        
        Args:
            docs: 文档列表
            field: 要提取的字段名
            
        Returns:
            包含文档和对应字段值的元组列表
        """
        valid_docs = []
        std_field = field_mapping_manager.get_standard_field_name(field)
        
        for doc in docs:
            try:
                doc_values = DocumentParser.parse_document(doc)
                if std_field in doc_values:
                    try:
                        value = float(doc_values[std_field])
                        valid_docs.append((doc, value))
                    except ValueError:
                        continue
            except Exception as e:
                logger.warning(f"Error extracting field value: {str(e)}")
                continue
        return valid_docs
    
    @staticmethod
    def extract_unique_categories(docs: List[Document]) -> Set[str]:
        """从文档中提取所有唯一的类别
        
        Args:
            docs: 文档列表
            
        Returns:
            唯一类别集合
        """
        categories = set()
        category_field = field_mapping_manager.get_fields_by_type(FieldType.CATEGORY)[0]
        
        for doc in docs:
            doc_values = DocumentParser.parse_document(doc)
            if category := doc_values.get(category_field):
                categories.add(category.strip())
        return categories

class BaseQueryProcessor(ABC):
    """查询处理器基类"""
    
    def __init__(self):
        """初始化查询处理器"""
        self.vectorstore = None
        
    @abstractmethod
    def can_process(self, query: str) -> bool:
        """判断是否可以处理该查询"""
        pass
        
    @abstractmethod
    def process(self, query: str, docs: List[Document]) -> Dict[str, Any]:
        """处理查询"""
        pass
        
    def _compute_field_similarity(self, text: str, field: str) -> float:
        """计算文本与字段的语义相似度"""
        try:
            if not self.vectorstore:
                return 0.0
            
            # 获取字段的所有可能表达
            field_info = field_mapping_manager.get_field_info(field)
            if not field_info:
                return 0.0
                
            # 计算与所有变体的最大相似度
            max_similarity = 0.0
            text_embedding = self.vectorstore.get_embedding(text)
            
            # 检查标准字段名
            field_embedding = self.vectorstore.get_embedding(field_info.name)
            max_similarity = self.vectorstore.compute_similarity(text_embedding, field_embedding)
            
            # 检查所有别名
            for alias in field_info.aliases:
                alias_embedding = self.vectorstore.get_embedding(alias)
                similarity = self.vectorstore.compute_similarity(text_embedding, alias_embedding)
                max_similarity = max(max_similarity, similarity)
            
            return max_similarity
            
        except Exception as e:
            logger.warning(f"Error computing field similarity: {str(e)}")
            return 0.0
            
    def _extract_field_from_query(self, query: str, field_type: Optional[FieldType] = None, threshold: float = 0.7) -> Optional[str]:
        """从查询中提取字段
        
        Args:
            query: 查询文本
            field_type: 可选的字段类型过滤
            threshold: 相似度阈值
            
        Returns:
            匹配的标准字段名
        """
        try:
            # 将查询分词
            words = list(jieba.cut(query))
            
            # 获取待匹配的字段列表
            if field_type:
                fields = field_mapping_manager.get_fields_by_type(field_type)
            else:
                fields = list(field_mapping_manager.get_all_field_names())
            
            # 对每个词计算与所有字段的相似度
            max_similarity = 0.0
            best_field = None
            
            for word in words:
                for field in fields:
                    similarity = self._compute_field_similarity(word, field)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_field = field
            
            if max_similarity >= threshold:
                logger.debug(f"Extracted field '{best_field}' with similarity {max_similarity:.4f}")
                return best_field
                
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting field: {str(e)}")
            return None

class ComparisonQueryProcessor(BaseQueryProcessor):
    """比较查询处理器"""
    
    def __init__(self):
        super().__init__()
        self.comparison_patterns = {
            "max": [r"最贵的", r"最高的", r"最多的", r"最大的"],
            "min": [r"最便宜的", r"最低的", r"最少的", r"最小的"]
        }
        
    def can_process(self, query: str) -> bool:
        """检查是否是比较类查询"""
        return any(pattern in query 
                  for patterns in self.comparison_patterns.values() 
                  for pattern in patterns)
        
    def process(self, query: str, docs: List[Document]) -> Dict[str, Any]:
        """处理比较类查询"""
        try:
            if not docs:
                return {"error": "No documents available for comparison"}
                
            # 提取比较字段（仅从数值类型字段中查找）
            field = self._extract_field_from_query(query, FieldType.NUMERIC)
            if not field:
                field = "price"  # 默认比较价格
                
            # 确定比较方向
            is_max = any(p in query for p in self.comparison_patterns["max"])
            
            # 提取字段值
            valid_docs = DocumentParser.extract_field_values(docs, field)
            
            if not valid_docs:
                return {"error": f"No valid documents found with {field} field"}
                
            # 排序并选择结果
            sorted_docs = sorted(valid_docs, key=lambda x: x[1], reverse=is_max)
            result_docs = [doc for doc, _ in sorted_docs[:3]]
            
            return {
                "docs": result_docs,
                "context": "\n".join(doc.page_content for doc in result_docs)
            }
            
        except Exception as e:
            return {"error": str(e)}

class CategoryFilterProcessor(BaseQueryProcessor):
    """类别过滤处理器"""
    
    def can_process(self, query: str) -> bool:
        """检查是否是类别过滤查询"""
        category_field = field_mapping_manager.get_fields_by_type(FieldType.CATEGORY)[0]
        field_info = field_mapping_manager.get_field_info(category_field)
        return any(keyword in query for keyword in field_info.aliases)
        
    def process(self, query: str, docs: List[Document]) -> Dict[str, Any]:
        """处理类别过滤查询"""
        try:
            if not docs:
                return {"error": "No documents available for filtering"}
                
            # 获取所有唯一类别
            categories = DocumentParser.extract_unique_categories(docs)
            if not categories:
                return {"error": "No categories found in documents"}
                
            # 提取查询中的类别
            words = list(jieba.cut(query))
            max_similarity = 0.0
            query_category = None
            
            for word in words:
                for category in categories:
                    similarity = self._compute_field_similarity(word, category)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        query_category = category
            
            if not query_category:
                return {"error": "No category found in query"}
                
            # 过滤文档
            filtered_docs = []
            for doc in docs:
                doc_values = DocumentParser.parse_document(doc)
                category_field = field_mapping_manager.get_fields_by_type(FieldType.CATEGORY)[0]
                doc_category = doc_values.get(category_field, "")
                if self._compute_field_similarity(query_category, doc_category) >= 0.7:
                    filtered_docs.append(doc)
            
            if not filtered_docs:
                return {"error": f"No documents found in category {query_category}"}
                
            return {
                "docs": filtered_docs,
                "context": "\n".join(doc.page_content for doc in filtered_docs[:3])
            }
            
        except Exception as e:
            return {"error": str(e)}

class QueryProcessorRegistry:
    """查询处理器注册中心"""
    
    def __init__(self):
        self._processors: List[BaseQueryProcessor] = []
        
    def register(self, processor: BaseQueryProcessor):
        """注册新的处理器"""
        self._processors.append(processor)
        
    def get_processor(self, query: str) -> Optional[BaseQueryProcessor]:
        """获取适合处理该查询的处理器"""
        for processor in self._processors:
            if processor.can_process(query):
                return processor
        return None 