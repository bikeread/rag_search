"""
Main entry point for the LLM Q&A system.
"""

import logging
import os
from src.monitoring.metrics import MetricsCollector
import uvicorn

# 配置日志
# 可通过环境变量设置日志级别:
# - LOG_LEVEL=DEBUG 启用完整调试日志，包括表格处理器的详细诊断信息
# - LOG_LEVEL=INFO 默认级别，只显示一般运行信息
# - LOG_LEVEL=WARNING 只显示警告和错误
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Start the Q&A system."""
    try:
        # 初始化指标收集器
        logger.info("Starting metrics collector...")
        metrics = MetricsCollector()
        
        # 启动API服务器
        host = os.getenv("API_HOST", "0.0.0.0")
        port = int(os.getenv("API_PORT", "8000"))
        reload = os.getenv("API_DEBUG", "").lower() == "true"
        
        logger.info(f"Starting API server on {host}:{port}...")
        
        if reload:
            # 使用模块导入字符串方式启动，支持热重载
            uvicorn.run(
                "src.api.app:app",
                host=host,
                port=port,
                reload=True
            )
        else:
            # 直接传递 app 实例启动
            from src.api.app import app
            uvicorn.run(
                app,
                host=host,
                port=port
            )
        
    except Exception as e:
        logger.error(f"Error starting system: {str(e)}")
        raise

if __name__ == "__main__":
    main() 