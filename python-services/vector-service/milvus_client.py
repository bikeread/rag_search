"""
Milvus客户端 - 引用shared实现
使用统一的Milvus客户端，避免重复代码
"""

# 导入shared版本的Milvus客户端
import sys
import os

# 添加shared目录到Python路径
shared_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'shared')
if shared_path not in sys.path:
    sys.path.insert(0, shared_path)

# 避免循环导入，直接导入shared模块
import shared.milvus_client as shared_milvus
MilvusClient = shared_milvus.MilvusClient
get_milvus_client = shared_milvus.get_milvus_client

# 为向后兼容性导出所有符号
__all__ = ['MilvusClient', 'get_milvus_client']