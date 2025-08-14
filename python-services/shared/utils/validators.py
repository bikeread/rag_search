"""
数据验证工具
"""

from typing import List, Dict, Any
import re
from .exceptions import DocumentProcessingError, QueryProcessingError


def validate_document(filename: str, content: bytes, mime_type: str) -> bool:
    """
    验证文档是否符合处理要求
    
    Args:
        filename: 文件名
        content: 文件内容
        mime_type: MIME类型
        
    Returns:
        bool: 验证结果
        
    Raises:
        DocumentProcessingError: 验证失败时抛出
    """
    # 验证文件名
    if not filename or len(filename.strip()) == 0:
        raise DocumentProcessingError("文件名不能为空")
    
    # 验证文件大小 (限制50MB)
    if len(content) > 50 * 1024 * 1024:
        raise DocumentProcessingError("文件大小超过50MB限制")
    
    # 验证文件类型
    allowed_types = [
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'application/msword',
        'text/plain',
        'text/markdown'
    ]
    
    if mime_type not in allowed_types:
        raise DocumentProcessingError(f"不支持的文件类型: {mime_type}")
    
    # 验证文件内容不为空
    if len(content) == 0:
        raise DocumentProcessingError("文件内容为空")
    
    return True


def validate_query(query: str, max_length: int = 1000) -> bool:
    """
    验证查询文本
    
    Args:
        query: 查询文本
        max_length: 最大长度限制
        
    Returns:
        bool: 验证结果
        
    Raises:
        QueryProcessingError: 验证失败时抛出
    """
    if not query or len(query.strip()) == 0:
        raise QueryProcessingError("查询内容不能为空")
    
    if len(query) > max_length:
        raise QueryProcessingError(f"查询长度超过{max_length}字符限制")
    
    # 基本安全检查 - 防止SQL注入等
    dangerous_patterns = [
        r"<script.*?>.*?</script>",
        r"javascript:",
        r"on\w+\s*="
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            raise QueryProcessingError("查询内容包含不安全字符")
    
    return True


def validate_vector(vector: List[float], expected_dim: int = 384) -> bool:
    """
    验证向量数据
    
    Args:
        vector: 向量数据
        expected_dim: 期望维度
        
    Returns:
        bool: 验证结果
    """
    if not isinstance(vector, list):
        return False
    
    if len(vector) != expected_dim:
        return False
    
    # 检查所有元素都是数字
    for val in vector:
        if not isinstance(val, (int, float)):
            return False
    
    return True


def validate_metadata(metadata: Dict[str, Any]) -> bool:
    """
    验证元数据格式
    
    Args:
        metadata: 元数据字典
        
    Returns:
        bool: 验证结果
    """
    if not isinstance(metadata, dict):
        return False
    
    # 检查必需字段
    required_fields = ['document_id', 'chunk_index']
    for field in required_fields:
        if field not in metadata:
            return False
    
    return True
