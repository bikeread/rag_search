#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
查询优化器示例脚本
展示如何使用查询优化功能提高检索命中率
"""

import os
import sys
import logging
from typing import List, Dict, Any

# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.rag.retriever import RAGRetriever, QueryOptimizer
from src.model.llm_factory import LLMFactory
from langchain_core.documents import Document

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def demo_query_optimizer():
    """展示查询优化器的用法"""
    logger.info("初始化查询优化器...")
    
    # 创建LLM实例
    llm = LLMFactory.create_llm()
    
    # 创建查询优化器
    optimizer = QueryOptimizer(llm=llm)
    
    # 测试查询
    test_queries = [
        "最贵的笔记本电脑多少钱?",
        "有哪些适合程序员使用的键盘?",
        "深度学习和机器学习的区别是什么?",
        "公司最近的财务状况怎么样?"
    ]
    
    for query in test_queries:
        logger.info(f"\n\n处理查询: '{query}'")
        
        # 1. 分解查询为更简单的子查询
        sub_queries = optimizer.decompose_query(query)
        logger.info(f"查询分解结果:")
        for i, sq in enumerate(sub_queries):
            logger.info(f"  子查询 {i+1}: {sq}")
        
        # 2. 扩展查询
        expanded_queries = optimizer.expand_query(query)
        logger.info(f"查询扩展结果:")
        for i, eq in enumerate(expanded_queries):
            logger.info(f"  扩展形式 {i+1}: {eq}")
        
        # 3. 提取关键词
        keywords = optimizer.extract_keywords(query)
        logger.info(f"关键词提取结果:")
        for i, kw in enumerate(keywords):
            logger.info(f"  关键词 {i+1}: {kw}")
            
        logger.info(f"" + "-" * 50)

def demo_enhanced_retrieval():
    """展示增强检索的用法"""
    logger.info("初始化RAG检索器...")
    
    # 创建RAG检索器
    retriever = RAGRetriever()
    
    # 通过示例文档演示
    create_sample_docs(retriever)
    
    # 测试查询
    test_query = "最贵的笔记本电脑多少钱?"
    logger.info(f"\n处理查询: '{test_query}'")
    
    # 常规检索
    logger.info("使用常规检索...")
    regular_docs = retriever.get_relevant_documents(test_query, k=3)
    logger.info(f"常规检索返回文档数: {len(regular_docs)}")
    if regular_docs:
        logger.info("常规检索结果示例:")
        for i, doc in enumerate(regular_docs[:2]):
            logger.info(f"  文档 {i+1}: {doc.page_content[:100]}...")
    
    # 增强检索
    logger.info("\n使用增强检索...")
    enhanced_docs = retriever.get_relevant_documents_enhanced(test_query, k=3)
    logger.info(f"增强检索返回文档数: {len(enhanced_docs)}")
    if enhanced_docs:
        logger.info("增强检索结果示例:")
        for i, doc in enumerate(enhanced_docs[:2]):
            logger.info(f"  文档 {i+1}: {doc.page_content[:100]}...")
    
    # 生成答案
    logger.info("\n使用增强检索生成答案...")
    result = retriever.generate_answer(test_query, use_enhanced_retrieval=True)
    logger.info(f"生成的答案:\n{result['answer']}")
    
    # 不使用增强检索生成答案
    logger.info("\n不使用增强检索生成答案...")
    result = retriever.generate_answer(test_query, use_enhanced_retrieval=False)
    logger.info(f"生成的答案:\n{result['answer']}")

def create_sample_docs(retriever: RAGRetriever):
    """创建一些示例文档并添加到检索器"""
    logger.info("添加示例文档到检索器...")
    
    sample_docs_text = [
        "产品名称: 苹果MacBook Pro 16 | 价格: 19999元 | 类别: 笔记本电脑 | 处理器: M1 Max | 内存: 32GB | 特点: 高性能专业笔记本电脑",
        "产品名称: 联想ThinkPad X1 Carbon | 价格: 12999元 | 类别: 笔记本电脑 | 处理器: Intel Core i7 | 内存: 16GB | 特点: 轻薄商务本",
        "产品名称: 戴尔XPS 17 | 价格: 24999元 | 类别: 笔记本电脑 | 处理器: Intel Core i9 | 内存: 64GB | 特点: 设计师专用高端笔记本",
        "高端笔记本电脑市场分析: 目前市场上最贵的笔记本电脑主要有戴尔Alienware系列、苹果MacBook Pro系列和微软Surface系列。其中售价最高的是定制版的戴尔Alienware，配置了顶级GPU和CPU，售价可达4万元以上。",
        "笔记本电脑价格区间: 入门级笔记本价格在3000-5000元，中端笔记本价格在6000-10000元，高端笔记本价格在10000-20000元，顶级旗舰笔记本价格在20000元以上。",
        "笔记本电脑选购指南: 选择笔记本电脑要考虑处理器、显卡、内存和存储容量等因素。不同用途的笔记本有不同的侧重点，游戏本注重显卡性能，商务本注重便携性和电池续航。",
        "键盘推荐: 程序员最适合使用的键盘包括机械键盘如HHKB Professional、Cherry MX系列和白光的FILCO。键盘选择要注重按键手感和键位布局。",
        "财务报表分析: 公司第三季度营收增长15%，净利润同比增长8%。销售额达到1.2亿元，毛利率保持在30%左右。现金流稳定，负债率有所下降。"
    ]
    
    # 转换为Document对象
    sample_docs = [Document(page_content=text) for text in sample_docs_text]
    
    # 添加到向量存储
    texts = [doc.page_content for doc in sample_docs]
    retriever.vector_store.add_documents(sample_docs)  # 直接传递Document对象
    logger.info(f"已添加 {len(texts)} 个示例文档到检索器")

if __name__ == "__main__":
    logger.info("启动查询优化器示例")
    
    # 演示查询优化器功能
    demo_query_optimizer()
    
    # 演示增强检索功能
    demo_enhanced_retrieval()
    
    logger.info("示例完成") 