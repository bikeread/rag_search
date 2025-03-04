from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class VectorStoreConfig:
    """
    向量存储配置。
    """
    store_type: str
    model_name: str
    extra_args: Optional[Dict[str, Any]] = None 