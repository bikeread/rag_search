"""
自定义异常类
"""

class RAGServiceError(Exception):
    """RAG服务基础异常"""
    def __init__(self, message: str, error_code: str = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "RAG_ERROR"


class MilvusConnectionError(RAGServiceError):
    """Milvus连接异常"""
    def __init__(self, message: str = "Milvus connection failed"):
        super().__init__(message, "MILVUS_CONNECTION_ERROR")


class VectorizationError(RAGServiceError):
    """向量化异常"""
    def __init__(self, message: str = "Vectorization failed"):
        super().__init__(message, "VECTORIZATION_ERROR")


class DocumentProcessingError(RAGServiceError):
    """文档处理异常"""
    def __init__(self, message: str = "Document processing failed"):
        super().__init__(message, "DOCUMENT_PROCESSING_ERROR")


class QueryProcessingError(RAGServiceError):
    """查询处理异常"""
    def __init__(self, message: str = "Query processing failed"):
        super().__init__(message, "QUERY_PROCESSING_ERROR")


class LLMServiceError(RAGServiceError):
    """LLM服务异常"""
    def __init__(self, message: str = "LLM service error"):
        super().__init__(message, "LLM_SERVICE_ERROR")
