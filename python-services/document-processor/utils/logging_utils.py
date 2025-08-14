"""
日志工具模块，提供日志装饰器和辅助函数。
"""
import functools
import logging
import time
import json
from typing import Any, Callable, Dict, List, Union
import inspect

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def log_args_and_result(func: Callable) -> Callable:
    """
    装饰器：记录函数的输入参数和返回结果
    
    Args:
        func (Callable): 被装饰的函数
        
    Returns:
        Callable: 装饰后的函数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_logger = logging.getLogger(func.__module__)
        func_name = func.__qualname__
        
        # 获取函数签名
        sig = inspect.signature(func)
        
        # 处理参数（转换不可序列化的对象）
        def safe_repr(obj):
            if hasattr(obj, '__dict__'):
                return f"{obj.__class__.__name__} object"
            elif isinstance(obj, (list, tuple)):
                return f"{type(obj).__name__} with {len(obj)} items"
            elif isinstance(obj, dict):
                return f"dict with {len(obj)} items"
            return repr(obj)
        
        # 构建参数字典
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # 过滤掉 self 参数
        arg_dict = {k: v for k, v in bound_args.arguments.items() if k != 'self'}
        
        # 将参数转换为字符串
        safe_args = {}
        for k, v in arg_dict.items():
            if k == 'query':
                safe_args[k] = v  # 保留完整查询
            elif isinstance(v, (str, int, float, bool, type(None))):
                safe_args[k] = v
            else:
                safe_args[k] = safe_repr(v)
        
        # 记录输入参数
        func_logger.info(f"调用 {func_name} - 参数: {safe_args}")
        
        # 执行函数
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        
        # 记录返回结果
        if result is None:
            func_logger.info(f"{func_name} 执行完成 - 耗时: {elapsed:.4f}秒 - 返回: None")
        elif isinstance(result, (str, int, float, bool)):
            func_logger.info(f"{func_name} 执行完成 - 耗时: {elapsed:.4f}秒 - 返回: {result}")
        elif isinstance(result, dict):
            # 对于字典类型，打印键和值的类型
            safe_result = {k: f"{type(v).__name__}[{len(v) if hasattr(v, '__len__') else '?'}]" 
                           for k, v in result.items()}
            func_logger.info(f"{func_name} 执行完成 - 耗时: {elapsed:.4f}秒 - 返回: {safe_result}")
        elif isinstance(result, (list, tuple)):
            func_logger.info(f"{func_name} 执行完成 - 耗时: {elapsed:.4f}秒 - 返回: {type(result).__name__}[{len(result)}]")
        else:
            func_logger.info(f"{func_name} 执行完成 - 耗时: {elapsed:.4f}秒 - 返回: {type(result).__name__} 对象")
        
        return result
    return wrapper 