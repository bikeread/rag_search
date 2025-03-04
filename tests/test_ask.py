import unittest
import logging
from src.rag.retriever import RAGRetriever
from src.data_processing.storage.config import VectorStoreConfig
from src.data_processing.storage.factory import VectorStoreFactory
from src.model.ollama_client import OllamaClient
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestAskAPI(unittest.TestCase):
    """测试问答功能"""

    def setUp(self):
        """测试前的准备工作"""
        # 配置向量存储
        config = VectorStoreConfig(
            store_type="milvus",
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            extra_args={
                "host": "localhost",
                "port": "19530",
                "collection_name": "laptop_collection"
            }
        )
        
        # 检查并创建集合
        self._ensure_collection_exists(config.extra_args["collection_name"])
        
        # 创建向量存储
        vector_store = VectorStoreFactory.create_store(config)
        
        # 创建语言模型客户端
        llm = OllamaClient(model_name="deepseek-r1:7b")
        
        # 创建RAG检索器
        self.rag_retriever = RAGRetriever(
            vector_store=vector_store,
            llm=llm,
            query_enhancement_enabled=True,
            chain_type="qa"
        )
        
        # 添加测试数据
        test_data = [
            {
                "content": "产品ID: P001\n产品名称: 联想 ThinkPad X1 Carbon\n类别: 笔记本电脑\n价格: 12999\n处理器: Intel i7-1165G7\n内存: 16GB\n硬盘: 1TB SSD\n显卡: Intel Iris Xe Graphics\n特点: 轻薄商务本",
                "metadata": {
                    "product_id": "P001",
                    "product_name": "联想 ThinkPad X1 Carbon",
                    "category": "笔记本电脑",
                    "price": 12999,
                    "processor": "Intel i7-1165G7",
                    "memory": "16GB",
                    "storage": "1TB SSD",
                    "graphics": "Intel Iris Xe Graphics",
                    "features": "轻薄商务本"
                }
            },
            {
                "content": "产品ID: P002\n产品名称: 华硕 ROG 魔霸新锐\n类别: 游戏本\n价格: 9999\n处理器: AMD Ryzen 7 5800H\n内存: 16GB\n硬盘: 512GB SSD\n显卡: NVIDIA RTX 3060\n特点: 游戏性能强劲",
                "metadata": {
                    "product_id": "P002",
                    "product_name": "华硕 ROG 魔霸新锐",
                    "category": "游戏本",
                    "price": 9999,
                    "processor": "AMD Ryzen 7 5800H",
                    "memory": "16GB",
                    "storage": "512GB SSD",
                    "graphics": "NVIDIA RTX 3060",
                    "features": "游戏性能强劲"
                }
            },
            {
                "content": "产品ID: P003\n产品名称: 戴尔 XPS 15\n类别: 笔记本电脑\n价格: 15999\n处理器: Intel i9-11900H\n内存: 32GB\n硬盘: 2TB SSD\n显卡: NVIDIA RTX 3050 Ti\n特点: 高性能创作本",
                "metadata": {
                    "product_id": "P003",
                    "product_name": "戴尔 XPS 15",
                    "category": "笔记本电脑",
                    "price": 15999,
                    "processor": "Intel i9-11900H",
                    "memory": "32GB",
                    "storage": "2TB SSD",
                    "graphics": "NVIDIA RTX 3050 Ti",
                    "features": "高性能创作本"
                }
            }
        ]
        
        # 添加数据到向量存储
        vector_store.add_documents(test_data)

    def _ensure_collection_exists(self, collection_name: str):
        """确保集合存在，如果不存在则创建"""
        try:
            # 连接到 Milvus
            connections.connect(
                alias="default",
                host="localhost",
                port="19530"
            )
            
            # 如果集合已存在，先删除
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                logger.info(f"已删除现有集合: {collection_name}")
            
            # 创建新集合
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            schema = CollectionSchema(fields=fields, description="Document collection")
            collection = Collection(collection_name, schema)
            
            # 创建索引
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            logger.info(f"成功创建集合: {collection_name}")
            
            # 加载集合
            collection.load()
            
        except Exception as e:
            logger.error(f"创建集合时发生错误: {str(e)}")
            raise

    def test_ask_question(self):
        """测试问答功能"""
        # 测试问题列表
        questions = [
            "最贵的笔记本电脑多少钱？",
            "有哪些游戏本推荐？"
        ]
        
        # 测试每个问题
        for question in questions:
            logger.info(f"\n开始测试问题: {question}")
            
            # 生成答案
            result = self.rag_retriever.generate_answer(question)
            
            # 记录结果
            logger.info(f"问题: {question}")
            logger.info(f"回答: {result.get('answer', '无答案')}")
            logger.info(f"参考来源:")
            for source in result.get("sources", []):
                logger.info(f"- {source}")
            logger.info("-" * 50)
            
            # 验证结果格式
            self.assertIsInstance(result, dict)
            self.assertIn("answer", result)
            self.assertIn("sources", result)
            self.assertIsInstance(result["answer"], str)
            self.assertIsInstance(result["sources"], list)

if __name__ == "__main__":
    unittest.main() 