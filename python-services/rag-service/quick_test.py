#!/usr/bin/env python3
"""
Cross-Encoder快速功能测试
"""

import asyncio
from main import CrossEncoderReranker

async def quick_test():
    print("🧪 Cross-Encoder快速功能测试")
    
    # 1. 测试初始化
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=4,
        cache_size=100
    )
    
    print(f"✅ 初始化完成，启用状态: {reranker.enabled}")
    
    if not reranker.enabled:
        print("⚠️  Cross-Encoder未启用，可能是依赖未安装")
        return
    
    # 2. 测试重排序功能
    test_docs = [
        {"id": "1", "text": "机器学习是人工智能的一个分支", "score": 0.8},
        {"id": "2", "text": "深度学习使用神经网络", "score": 0.7},
        {"id": "3", "text": "今天天气很好", "score": 0.6}
    ]
    
    query = "什么是机器学习？"
    
    print(f"🔍 测试查询: {query}")
    print(f"📄 候选文档数: {len(test_docs)}")
    
    try:
        reranked = await reranker.rerank_documents(query, test_docs, top_k=2)
        print(f"✅ 重排序完成，返回文档数: {len(reranked)}")
        
        for i, doc in enumerate(reranked):
            print(f"   排名 {i+1}: ID={doc['id']}, "
                  f"原始分数={doc.get('original_score', 'N/A')}, "
                  f"重排序分数={doc.get('rerank_score', 'N/A'):.3f}")
        
        # 3. 测试性能统计
        stats = reranker.get_performance_stats()
        print(f"📊 性能统计: {stats}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    print("✅ 快速测试完成")

if __name__ == "__main__":
    asyncio.run(quick_test())