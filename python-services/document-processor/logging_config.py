"""
日志配置模块
统一的日志记录配置
"""

import logging
import logging.config
import json
import sys
from datetime import datetime
from typing import Dict, Any
from config import settings


class JSONFormatter(logging.Formatter):
    """JSON格式的日志格式化器"""
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为JSON"""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # 添加异常信息
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # 添加额外的字段
        if hasattr(record, 'service_name'):
            log_entry["service"] = record.service_name
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        if hasattr(record, 'execution_time'):
            log_entry["execution_time"] = record.execution_time
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = record.user_id
            
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(service_name: str = "rag-service") -> logging.Logger:
    """
    设置日志配置
    
    Args:
        service_name: 服务名称
        
    Returns:
        logging.Logger: 配置好的日志器
    """
    
    # 日志配置字典
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": settings.log_format
            },
            "json": {
                "()": JSONFormatter
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.log_level,
                "formatter": "json" if settings.enable_json_logs else "standard",
                "stream": sys.stdout
            }
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["console"],
                "level": settings.log_level,
                "propagate": False
            },
            "uvicorn": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False
            },
            "fastapi": {
                "handlers": ["console"],
                "level": "INFO", 
                "propagate": False
            }
        }
    }
    
    # 应用配置
    logging.config.dictConfig(log_config)
    
    # 获取logger并添加服务名称
    logger = logging.getLogger(service_name)
    
    # 创建适配器添加服务名称
    class ServiceAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            return msg, kwargs
            
        def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False):
            if extra is None:
                extra = {}
            extra['service_name'] = service_name
            super()._log(level, msg, args, exc_info, extra, stack_info)
    
    return ServiceAdapter(logger, {})


def get_logger(name: str = None) -> logging.Logger:
    """
    获取日志器
    
    Args:
        name: 日志器名称，默认为调用模块名
        
    Returns:
        logging.Logger: 日志器实例
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return logging.getLogger(name)


class RequestLogger:
    """请求日志记录器"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_request_start(self, request_id: str, method: str, path: str, **kwargs):
        """记录请求开始"""
        self.logger.info(
            f"Request started: {method} {path}",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                **kwargs
            }
        )
    
    def log_request_end(self, request_id: str, status_code: int, execution_time: float, **kwargs):
        """记录请求结束"""
        self.logger.info(
            f"Request completed: {status_code} ({execution_time:.3f}s)",
            extra={
                "request_id": request_id,
                "status_code": status_code,
                "execution_time": execution_time,
                **kwargs
            }
        )
    
    def log_request_error(self, request_id: str, error: Exception, **kwargs):
        """记录请求错误"""
        self.logger.error(
            f"Request failed: {str(error)}",
            extra={
                "request_id": request_id,
                "error_type": type(error).__name__,
                **kwargs
            },
            exc_info=True
        )


# 预配置的日志器
def get_service_logger(service_name: str) -> logging.Logger:
    """获取服务专用日志器"""
    return setup_logging(service_name)
