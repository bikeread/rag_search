#!/usr/bin/env python3
"""
Cross-Encoder重排序集成测试

验证Cross-Encoder功能的正确集成和性能表现
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any

# 测试配置
RAG_SERVICE_URL = "http://localhost:8001"

async def test_cross_encoder_integration():
    """测试Cross-Encoder重排序集成"""
    
    print("🧪 Cross-Encoder重排序集成测试开始")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        
        # 1. 检查服务健康状态和Cross-Encoder可用性
        print("1️⃣ 检查服务健康状态...")
        try:
            async with session.get(f"{RAG_SERVICE_URL}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"✅ 服务状态: {health_data.get('status', 'unknown')}")
                    
                    # 检查Cross-Encoder功能
                    cross_encoder_features = health_data.get('cross_encoder_features', {})
                    enabled = cross_encoder_features.get('enabled', False)
                    available = cross_encoder_features.get('reranking_available', False)
                    model_name = cross_encoder_features.get('model_name', 'N/A')
                    
                    print(f"🎯 Cross-Encoder状态:")
                    print(f"   - 依赖可用: {available}")
                    print(f"   - 功能启用: {enabled}")
                    print(f"   - 模型名称: {model_name}")
                    
                    if not available:
                        print("⚠️  Cross-Encoder依赖不可用，请安装: pip install sentence-transformers torch")
                        return
                    
                else:
                    print(f"❌ 服务健康检查失败: {response.status}")
                    return
                    
        except Exception as e:
            print(f"❌ 无法连接到RAG服务: {e}")
            return
        
        # 2. 测试不启用重排序的基准查询
        print("\n2️⃣ 执行基准查询（不启用重排序）...")
        baseline_query = {
            "query": "什么是机器学习？",
            "top_k": 3,
            "retrieval_mode": "hybrid",
            "enable_reranking": False,
            "enable_multiround": False,
            "enable_quality_control": False
        }
        
        baseline_time, baseline_results = await execute_query(session, baseline_query)
        print(f"⏱️  基准查询耗时: {baseline_time:.3f}秒")
        if baseline_results:
            print(f"📊 基准结果数量: {len(baseline_results.get('sources', []))}")
        
        # 3. 测试启用重排序的查询
        print("\n3️⃣ 执行重排序查询（启用Cross-Encoder）...")
        rerank_query = {
            "query": "什么是机器学习？",
            "top_k": 3,
            "rerank_top_k": 10,
            "retrieval_mode": "hybrid",
            "enable_reranking": True,
            "enable_multiround": False,
            "enable_quality_control": False
        }
        
        rerank_time, rerank_results = await execute_query(session, rerank_query)
        print(f"⏱️  重排序查询耗时: {rerank_time:.3f}秒")
        if rerank_results:
            print(f"📊 重排序结果数量: {len(rerank_results.get('sources', []))}")
            
            # 分析重排序信息
            metadata = rerank_results.get('metadata', {})
            rerank_info = metadata.get('reranking_info', {})
            
            print(f"🎯 重排序分析:")
            print(f"   - 重排序执行: {rerank_info.get('reranked', False)}")
            print(f"   - 候选文档数: {rerank_info.get('candidates_count', 'N/A')}")
            print(f"   - 最终文档数: {rerank_info.get('final_count', 'N/A')}")
            
            if 'performance' in rerank_info:
                perf = rerank_info['performance']
                print(f"   - 平均处理时间: {perf.get('avg_time_per_request', 'N/A')}秒")
                print(f"   - 缓存命中率: {perf.get('cache_hit_rate', 'N/A')}")
        
        # 4. 性能对比分析
        print("\n4️⃣ 性能对比分析...")
        if baseline_time and rerank_time:
            time_overhead = rerank_time - baseline_time
            overhead_percentage = (time_overhead / baseline_time) * 100
            
            print(f"⚡ 重排序性能开销:")
            print(f"   - 额外耗时: {time_overhead:.3f}秒")
            print(f"   - 开销百分比: {overhead_percentage:.1f}%")
            
            if time_overhead < 0.2:  # 目标<200ms
                print(f"✅ 性能开销符合要求 (<200ms)")
            else:
                print(f"⚠️  性能开销超出目标 (>200ms)")
        
        # 5. 获取详细统计信息
        print("\n5️⃣ 获取Cross-Encoder详细统计...")
        try:
            async with session.get(f"{RAG_SERVICE_URL}/stats/cross-encoder") as response:
                if response.status == 200:
                    stats_data = await response.json()
                    stats = stats_data.get('stats', {})
                    model_info = stats_data.get('model_info', {})
                    
                    print(f"📈 Cross-Encoder统计信息:")
                    print(f"   - 总处理次数: {stats.get('rerank_count', 0)}")
                    print(f"   - 总处理时间: {stats.get('total_time', 0)}秒")
                    print(f"   - 平均处理时间: {stats.get('avg_time_per_request', 0)}秒")
                    print(f"   - 缓存命中率: {stats.get('cache_hit_rate', 0)}")
                    print(f"   - 缓存大小: {stats.get('cache_size', 0)}")
                    
                    if model_info:
                        print(f"🤖 模型配置:")
                        print(f"   - 模型名称: {model_info.get('name', 'N/A')}")
                        print(f"   - 最大长度: {model_info.get('max_length', 'N/A')}")
                        print(f"   - 批处理大小: {model_info.get('batch_size', 'N/A')}")
        
        except Exception as e:
            print(f"⚠️  获取统计信息失败: {e}")
        
        # 6. 测试批量查询以评估缓存效果
        print("\n6️⃣ 测试批量查询（评估缓存效果）...")
        test_queries = [
            "什么是人工智能？",
            "机器学习的主要类型有哪些？",
            "深度学习和传统机器学习的区别？"
        ]
        
        batch_times = []
        for i, test_query in enumerate(test_queries):
            query_config = {
                "query": test_query,
                "top_k": 3,
                "rerank_top_k": 8,
                "enable_reranking": True,
                "enable_multiround": False,
                "enable_quality_control": False
            }
            
            start_time = time.time()
            _, result = await execute_query(session, query_config)
            elapsed = time.time() - start_time
            batch_times.append(elapsed)
            
            print(f"   查询 {i+1}: {elapsed:.3f}秒")
        
        if batch_times:
            avg_batch_time = sum(batch_times) / len(batch_times)
            print(f"📊 批量查询平均耗时: {avg_batch_time:.3f}秒")
    
    print("\n" + "=" * 60)
    print("✅ Cross-Encoder重排序集成测试完成")

async def execute_query(session: aiohttp.ClientSession, query_config: Dict[str, Any]):
    """执行查询并返回耗时和结果"""
    try:
        start_time = time.time()
        
        async with session.post(
            f"{RAG_SERVICE_URL}/query",
            json=query_config,
            headers={"Content-Type": "application/json"}
        ) as response:
            elapsed_time = time.time() - start_time
            
            if response.status == 200:
                result = await response.json()
                return elapsed_time, result
            else:
                error_text = await response.text()
                print(f"❌ 查询失败 ({response.status}): {error_text}")
                return elapsed_time, None
                
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ 查询异常: {e}")
        return elapsed_time, None

if __name__ == "__main__":
    print("🚀 启动Cross-Encoder重排序集成测试...")
    asyncio.run(test_cross_encoder_integration())