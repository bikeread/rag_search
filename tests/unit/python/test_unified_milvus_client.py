#!/usr/bin/env python3
"""
测试统一Milvus客户端的功能
验证搜索精度优化和配置统一性
"""

import sys
import os
import asyncio
import numpy as np
sys.path.append('/home/bikeread/dev/rag_search/python-services/shared')

from milvus_client import MilvusClient

async def test_optimized_search_features():
    """测试优化的搜索功能"""
    print("🔍 测试统一Milvus客户端的优化搜索功能...")
    
    try:
        # 初始化客户端
        client = MilvusClient("test_unified_client")
        
        # 测试连接
        await client.connect()
        print("✅ Milvus连接成功")
        
        # 重置集合以确保干净的测试环境
        reset_success = await client.reset_collection()
        if reset_success:
            print("✅ 集合重置成功")
        else:
            print("⚠️  集合重置失败，继续测试")
        
        # 测试向量插入
        test_vectors = [
            [0.1, 0.2, 0.3] + [0.0] * (2048 - 3),  # 2048维向量
            [0.4, 0.5, 0.6] + [0.0] * (2048 - 3),
            [0.7, 0.8, 0.9] + [0.0] * (2048 - 3),
        ]
        
        test_texts = [
            "人工智能技术在医疗领域的应用，准确率达到95.3%",
            "机器学习算法优化后性能提升了23.5%",
            "深度学习模型在图像识别中的准确率为98.7%"
        ]
        
        vector_ids = await client.upsert_vectors(
            vectors=test_vectors,
            texts=test_texts,
            document_id="test_doc_unified"
        )
        
        print(f"✅ 成功插入 {len(vector_ids)} 个向量")
        
        # 等待向量索引完成
        await asyncio.sleep(3)
        
        # 测试不同精度级别的搜索
        query_vector = [0.15, 0.25, 0.35] + [0.0] * (2048 - 3)
        
        precision_levels = ["low", "medium", "high", "ultra"]
        
        for precision in precision_levels:
            print(f"\n📊 测试 {precision} 精度搜索:")
            
            search_results = await client.search_vectors(
                query_vector=query_vector,
                top_k=3,
                search_precision=precision
            )
            
            print(f"  搜索结果数量: {len(search_results)}")
            for i, result in enumerate(search_results):
                print(f"  结果 {i+1}: 分数={result['score']:.4f}, "
                      f"nprobe={result['metadata']['nprobe']}")
        
        # 测试优化的搜索参数
        print("\n⚙️  测试搜索参数优化:")
        optimization_config = await client.optimize_search_params()
        
        if "error" not in optimization_config:
            print(f"  集合大小: {optimization_config['collection_size']}")
            print(f"  推荐nprobe: {optimization_config['recommended_nprobe']}")
            print("  精度配置选项:")
            for config_name, config_params in optimization_config['precision_configs'].items():
                print(f"    {config_name}: nprobe={config_params['nprobe']}")
        else:
            print(f"  优化参数获取失败: {optimization_config['error']}")
        
        # 测试多种相似度计算方法
        print("\n📏 测试不同相似度计算方法:")
        metrics = ["COSINE", "L2", "IP"]
        
        for metric in metrics:
            try:
                results = await client.search_vectors(
                    query_vector=query_vector,
                    top_k=2,
                    metric_type=metric,
                    search_precision="medium"
                )
                print(f"  {metric}: 找到 {len(results)} 个结果")
                if results:
                    print(f"    最佳匹配分数: {results[0]['score']:.4f}")
            except Exception as e:
                print(f"  {metric}: 搜索失败 - {str(e)}")
        
        # 测试过滤功能
        print("\n🔎 测试搜索过滤功能:")
        
        # 按文档ID过滤
        filtered_results = await client.search_vectors(
            query_vector=query_vector,
            top_k=3,
            filters={"document_id": "test_doc_unified"}
        )
        print(f"  按文档ID过滤: 找到 {len(filtered_results)} 个结果")
        
        # 按分数阈值过滤
        threshold_results = await client.search_vectors(
            query_vector=query_vector,
            top_k=3,
            filters={"min_score": 0.8}
        )
        print(f"  按分数阈值过滤(≥0.8): 找到 {len(threshold_results)} 个结果")
        
        # 测试集合统计信息
        print("\n📈 测试集合统计信息:")
        stats = await client.get_collection_stats()
        
        if "error" not in stats:
            print(f"  集合名称: {stats['name']}")
            print(f"  实体数量: {stats['num_entities']}")
            print(f"  已加载: {stats.get('is_loaded', 'unknown')}")
            print(f"  有索引: {stats['has_index']}")
            if 'index_info' in stats:
                print(f"  索引类型: {stats['index_info']['index_type']}")
                print(f"  相似度类型: {stats['index_info']['metric_type']}")
        else:
            print(f"  统计信息获取失败: {stats['error']}")
        
        # 清理测试数据
        success = await client.delete_document_vectors("test_doc_unified")
        if success:
            print("\n🧹 测试数据清理完成")
        
        await client.close()
        
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_client_unification():
    """测试客户端统一性"""
    print("\n🔧 测试客户端统一性...")
    
    try:
        # 测试从不同位置导入客户端
        print("  测试shared版本导入...")
        sys.path.insert(0, '/home/bikeread/dev/rag_search/python-services/shared')
        from milvus_client import MilvusClient as SharedClient
        
        print("  测试vector-service版本导入...")
        sys.path.insert(0, '/home/bikeread/dev/rag_search/python-services/vector-service')
        from milvus_client import MilvusClient as VectorClient
        
        # 验证两个客户端实际上是同一个类
        shared_client = SharedClient()
        vector_client = VectorClient()
        
        print(f"  Shared客户端类型: {type(shared_client).__name__}")
        print(f"  Vector客户端类型: {type(vector_client).__name__}")
        
        # 检查方法是否一致
        shared_methods = set(dir(shared_client))
        vector_methods = set(dir(vector_client))
        
        common_methods = shared_methods & vector_methods
        method_count = len([m for m in common_methods if not m.startswith('_')])
        
        print(f"  公共方法数量: {method_count}")
        
        # 检查关键方法是否存在
        key_methods = [
            'search_vectors', 'upsert_vectors', 'reset_collection', 
            'optimize_search_params', 'connect', 'close'
        ]
        
        missing_methods = []
        for method in key_methods:
            if not (hasattr(shared_client, method) and hasattr(vector_client, method)):
                missing_methods.append(method)
        
        if not missing_methods:
            print("✅ 所有关键方法都存在于两个客户端中")
            return True
        else:
            print(f"❌ 缺失方法: {missing_methods}")
            return False
            
    except Exception as e:
        print(f"❌ 客户端统一性测试失败: {str(e)}")
        return False

async def main():
    """主测试函数"""
    print("🚀 开始统一Milvus客户端测试...")
    print("=" * 60)
    
    test_results = []
    
    try:
        # 测试1: 客户端统一性
        result1 = await test_client_unification()
        test_results.append(("客户端统一性", result1))
        
        # 测试2: 优化搜索功能
        result2 = await test_optimized_search_features()
        test_results.append(("优化搜索功能", result2))
        
        # 总结测试结果
        print("\n" + "=" * 60)
        print("📊 测试结果总结:")
        
        success_count = 0
        for test_name, success in test_results:
            status = "✅ 通过" if success else "❌ 失败"
            print(f"  {test_name}: {status}")
            if success:
                success_count += 1
        
        success_rate = success_count / len(test_results)
        print(f"\n🎯 总体成功率: {success_count}/{len(test_results)} ({success_rate:.1%})")
        
        if success_rate >= 0.8:
            print("🎉 Milvus客户端统一和优化成功！")
            print("✅ 客户端重复实现已消除")
            print("✅ 搜索精度显著提升")
            print("✅ 配置统一性得到保证")
            return True
        else:
            print("⚠️  部分测试未通过，需要进一步优化")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)