"""
共享组件包
提供所有Python微服务使用的共享功能
"""

from .config import settings
from .milvus_client import MilvusClient
from .rabbit_client import RabbitMQClient

__version__ = "1.0.0"
__all__ = ["settings", "MilvusClient", "RabbitMQClient"]
