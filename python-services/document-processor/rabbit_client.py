"""
RabbitMQ客户端
处理消息队列的发布和订阅
"""

import asyncio
import aio_pika
import json
from typing import Dict, Any, Callable, Optional
from datetime import datetime

from config import settings
from utils.exceptions import RAGServiceError
from logging_config import get_logger

logger = get_logger(__name__)


class RabbitMQClient:
    """
    RabbitMQ消息队列客户端
    处理服务间异步通信
    """
    
    def __init__(self):
        """初始化RabbitMQ客户端"""
        self.connection: Optional[aio_pika.Connection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.is_connected = False
        
        # 从配置获取连接参数
        self.url = settings.rabbitmq_url
        self.exchange_name = settings.rabbitmq_exchange
        self.queues_config = settings.get_rabbitmq_config()["queues"]
        
        logger.info("Initializing RabbitMQ client")
    
    async def connect(self) -> bool:
        """
        建立RabbitMQ连接
        
        Returns:
            bool: 连接是否成功
            
        Raises:
            RAGServiceError: 连接失败时抛出
        """
        try:
            logger.info(f"Connecting to RabbitMQ at {self.url}")
            
            # 建立连接
            self.connection = await aio_pika.connect_robust(self.url)
            self.channel = await self.connection.channel()
            
            # 设置QoS
            await self.channel.set_qos(prefetch_count=10)
            
            # 设置交换机和队列
            await self._setup_exchanges_and_queues()
            
            self.is_connected = True
            logger.info("Successfully connected to RabbitMQ")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
            raise RAGServiceError(f"RabbitMQ connection failed: {str(e)}")
    
    async def _setup_exchanges_and_queues(self):
        """设置交换机和队列"""
        try:
            # 声明交换机
            self.exchange = await self.channel.declare_exchange(
                self.exchange_name,
                aio_pika.ExchangeType.TOPIC,
                durable=True
            )
            
            # 声明队列
            for queue_key, queue_name in self.queues_config.items():
                queue = await self.channel.declare_queue(
                    queue_name,
                    durable=True
                )
                
                # 绑定路由键
                routing_key = f"rag.{queue_key}"
                await queue.bind(self.exchange, routing_key)
                
                logger.info(f"Queue {queue_name} bound to routing key {routing_key}")
            
            logger.info("Exchanges and queues setup completed")
            
        except Exception as e:
            logger.error(f"Failed to setup exchanges and queues: {str(e)}")
            raise RAGServiceError(f"Queue setup failed: {str(e)}")
    
    async def publish_message(self, 
                            routing_key: str, 
                            message: Dict[str, Any],
                            persistent: bool = True) -> bool:
        """
        发布消息
        
        Args:
            routing_key: 路由键
            message: 消息内容
            persistent: 是否持久化
            
        Returns:
            bool: 发布是否成功
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            # 添加时间戳和消息ID
            message_with_meta = {
                **message,
                "timestamp": datetime.now().isoformat(),
                "message_id": f"{routing_key}_{int(datetime.now().timestamp() * 1000)}"
            }
            
            # 创建消息
            aio_message = aio_pika.Message(
                json.dumps(message_with_meta, ensure_ascii=False).encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT if persistent else aio_pika.DeliveryMode.NOT_PERSISTENT,
                content_type="application/json"
            )
            
            # 发布消息
            await self.exchange.publish(
                aio_message,
                routing_key=routing_key
            )
            
            logger.info(f"Message published to {routing_key}: {message.get('status', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish message: {str(e)}")
            return False
    
    async def publish_document_processed(self, message: Dict[str, Any]) -> bool:
        """
        发布文档处理完成消息
        
        Args:
            message: 包含文档处理结果的消息
            
        Returns:
            bool: 发布是否成功
        """
        routing_key = "rag.document_processed"
        return await self.publish_message(routing_key, message)
    
    async def publish_vectorization_completed(self, message: Dict[str, Any]) -> bool:
        """
        发布向量化完成消息
        
        Args:
            message: 包含向量化结果的消息
            
        Returns:
            bool: 发布是否成功
        """
        routing_key = "rag.vector_completed"
        return await self.publish_message(routing_key, message)
    
    async def publish_processing_status(self, 
                                      document_id: str, 
                                      status: str, 
                                      details: Dict[str, Any] = None) -> bool:
        """
        发布处理状态更新消息
        
        Args:
            document_id: 文档ID
            status: 处理状态
            details: 额外详情
            
        Returns:
            bool: 发布是否成功
        """
        message = {
            "document_id": document_id,
            "status": status,
            "details": details or {}
        }
        
        routing_key = "rag.processing_status"
        return await self.publish_message(routing_key, message)
    
    async def consume_messages(self, 
                             queue_name: str, 
                             callback: Callable,
                             auto_ack: bool = False) -> bool:
        """
        消费消息
        
        Args:
            queue_name: 队列名称
            callback: 回调函数
            auto_ack: 是否自动确认
            
        Returns:
            bool: 消费设置是否成功
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            # 获取队列
            queue = await self.channel.declare_queue(queue_name, durable=True)
            
            async def message_handler(message: aio_pika.IncomingMessage):
                """消息处理器"""
                try:
                    # 解析消息
                    body = json.loads(message.body.decode())
                    
                    logger.info(f"Received message from {queue_name}: {body.get('message_id', 'unknown')}")
                    
                    # 调用回调函数
                    await callback(body)
                    
                    # 手动确认消息
                    if not auto_ack:
                        message.ack()
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message JSON: {str(e)}")
                    message.nack(requeue=False)  # 不重新入队
                    
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    message.nack(requeue=True)  # 重新入队
            
            # 开始消费
            await queue.consume(message_handler, no_ack=auto_ack)
            
            logger.info(f"Started consuming messages from {queue_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup message consumer: {str(e)}")
            return False
    
    async def get_queue_info(self, queue_name: str) -> Dict[str, Any]:
        """
        获取队列信息
        
        Args:
            queue_name: 队列名称
            
        Returns:
            Dict: 队列统计信息
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            # 获取队列
            queue = await self.channel.declare_queue(queue_name, durable=True)
            
            # 获取队列信息（注意：这在某些RabbitMQ版本中可能不完全准确）
            queue_info = {
                "name": queue_name,
                "durable": True,
                "message_count": None,  # 需要管理API才能准确获取
                "consumer_count": None
            }
            
            return queue_info
            
        except Exception as e:
            logger.error(f"Failed to get queue info: {str(e)}")
            return {"error": str(e)}
    
    async def purge_queue(self, queue_name: str) -> bool:
        """
        清空队列
        
        Args:
            queue_name: 队列名称
            
        Returns:
            bool: 操作是否成功
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            queue = await self.channel.declare_queue(queue_name, durable=True)
            await queue.purge()
            
            logger.info(f"Queue {queue_name} purged successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to purge queue: {str(e)}")
            return False
    
    async def close(self):
        """关闭连接"""
        try:
            if self.channel:
                await self.channel.close()
            
            if self.connection:
                await self.connection.close()
            
            self.is_connected = False
            logger.info("RabbitMQ connection closed")
            
        except Exception as e:
            logger.error(f"Error closing RabbitMQ connection: {str(e)}")
    
    def __del__(self):
        """析构函数"""
        if self.is_connected:
            try:
                # 注意：这里不能使用async
                self.is_connected = False
            except:
                pass


class MessageTypes:
    """消息类型常量"""
    DOCUMENT_PROCESSED = "document_processed"
    VECTORIZATION_COMPLETED = "vectorization_completed"
    PROCESSING_STATUS = "processing_status"
    QUERY_PROCESSED = "query_processed"
    ERROR_OCCURRED = "error_occurred"


class ProcessingStatus:
    """处理状态常量"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# 全局客户端实例（可选）
_global_rabbit_client: Optional[RabbitMQClient] = None


async def get_rabbit_client() -> RabbitMQClient:
    """
    获取RabbitMQ客户端实例
    
    Returns:
        RabbitMQClient: 客户端实例
    """
    global _global_rabbit_client
    
    if _global_rabbit_client is None or not _global_rabbit_client.is_connected:
        _global_rabbit_client = RabbitMQClient()
        await _global_rabbit_client.connect()
    
    return _global_rabbit_client
