"""
Configuration management.

This module manages all configuration settings for the application.
Configuration values are loaded from environment variables with sensible defaults.
Detailed configuration descriptions can be found in .env.example
"""

import os
from pathlib import Path
from typing import Union
from dotenv import load_dotenv, find_dotenv

# 基础路径配置
ROOT_DIR = Path(__file__).parent.parent
BASE_DIR = str(ROOT_DIR)

def get_path(relative_path: Union[str, Path]) -> str:
    """Convert relative path to absolute path based on ROOT_DIR"""
    return str(ROOT_DIR / relative_path)

# 加载环境变量，优先使用项目根目录下的 .env 文件
env_path = find_dotenv(usecwd=True) or str(ROOT_DIR / '.env')
load_dotenv(dotenv_path=env_path, override=True)

def get_env_value(key: str, default: str = "") -> str:
    """安全地获取环境变量值，移除注释和空白"""
    value = os.getenv(key, default)
    if value:
        value = value.split('#')[0].strip()
    return value or default

# LLM Settings
DEFAULT_HF_MODEL = "microsoft/phi-2"
LLM_PROVIDER = get_env_value("LLM_PROVIDER", "huggingface")
LLM_MODEL = get_env_value("LLM_MODEL", DEFAULT_HF_MODEL)
LLM_TEMPERATURE = float(get_env_value("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(get_env_value("LLM_MAX_TOKENS", "2048"))

# Ollama Settings
OLLAMA_BASE_URL = get_env_value("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = int(get_env_value("OLLAMA_TIMEOUT", "120"))

# Vector Database Settings
VECTOR_DB_TYPE = get_env_value("VECTOR_DB_TYPE", "faiss")
VECTOR_DB_PATH = get_path(get_env_value("VECTOR_DB_PATH", "data/vector_store.faiss"))
EMBEDDING_MODEL = get_env_value("EMBEDDING_MODEL", "BAAI/bge-m3")

# Milvus Settings
MILVUS_HOST = get_env_value("MILVUS_HOST", "localhost")
MILVUS_PORT = int(get_env_value("MILVUS_PORT", "19530"))
MILVUS_COLLECTION = get_env_value("MILVUS_COLLECTION", "document_store")
MILVUS_TIMEOUT = int(get_env_value("MILVUS_TIMEOUT", "30"))
MILVUS_USE_ALIASES = get_env_value("MILVUS_USE_ALIASES", "false").lower() == "true"

# RAG Settings
CHUNK_SIZE = int(get_env_value("CHUNK_SIZE", "1024"))
CHUNK_OVERLAP = int(get_env_value("CHUNK_OVERLAP", "200"))
TOP_K_RESULTS = int(get_env_value("TOP_K_RESULTS", "3"))
SEARCH_MULTIPLIER = int(get_env_value("SEARCH_MULTIPLIER", "2"))
MIN_SIMILARITY_SCORE = float(get_env_value("MIN_SIMILARITY_SCORE", "0.05"))

# Vector Index Settings
USE_IVF_INDEX = get_env_value("USE_IVF_INDEX", "false").lower() == "true"
IVF_NLIST = int(get_env_value("IVF_NLIST", "100"))
IVF_NPROBE = int(get_env_value("IVF_NPROBE", "10"))

# API Settings
API_HOST = get_env_value("API_HOST", "0.0.0.0")
API_PORT = int(get_env_value("API_PORT", "8000"))
API_DEBUG = get_env_value("API_DEBUG", "False").lower() == "true"

# Monitoring Settings
PROMETHEUS_PORT = int(get_env_value("PROMETHEUS_PORT", "9090"))
LOG_LEVEL = get_env_value("LOG_LEVEL", "INFO")

# Logging Settings
LOG_DIR = get_path('logs')
LOG_FILE = get_path('logs/vector_store.log')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# Document Processing Settings
DEFAULT_ENCODING = get_env_value("DEFAULT_ENCODING", "utf-8")
DEFAULT_LANGUAGE = get_env_value("DEFAULT_LANGUAGE", "zh")

# Unstructured Settings
UNSTRUCTURED_STRATEGY = get_env_value("UNSTRUCTURED_STRATEGY", "fast")
UNSTRUCTURED_OCR_LANGUAGES = get_env_value("UNSTRUCTURED_OCR_LANGUAGES", "eng,chi_sim").split(",")
UNSTRUCTURED_ENABLE_OCR = get_env_value("UNSTRUCTURED_ENABLE_OCR", "true").lower() == "true"
UNSTRUCTURED_PDF_OCR_THRESHOLD = float(get_env_value("UNSTRUCTURED_PDF_OCR_THRESHOLD", "0.8"))
UNSTRUCTURED_IMAGE_MIN_SIZE = int(get_env_value("UNSTRUCTURED_IMAGE_MIN_SIZE", "100"))
UNSTRUCTURED_HI_RES_MODEL = get_env_value("UNSTRUCTURED_HI_RES_MODEL", None) 