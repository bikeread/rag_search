#!/usr/bin/env python3
"""
TF-IDF向量化器优化测试脚本
验证新的向量化配置对搜索匹配度的提升效果
"""

import asyncio
import httpx
import json
import time
from typing import List, Dict, Any

class VectorOptimizationTester:
    def __init__(self, vector_service_url: str = "http://localhost:8002"):
        self.vector_service_url = vector_service_url
        
    async def test_vectorizer_config(self):
        """测试向量化器配置"""
        print("🔬 开始测试TF-IDF向量化器优化...")
        
        # 测试文档
        test_documents = [
            "机器学习是人工智能的重要分支，通过数据训练模型",
            "深度学习使用神经网络进行复杂的模式识别",
            "自然语言处理是AI在文本理解方面的应用",
            "计算机视觉让机器能够理解和分析图像内容",
            "人工智能正在改变我们的生活和工作方式"
        ]
        
        # 测试查询
        test_queries = [
            "什么是机器学习？",
            "神经网络如何工作？", 
            "AI在文本处理的应用",
            "图像识别技术",
            "AI对社会的影响"
        ]
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 1. 重置Milvus集合
            print("📋 重置Milvus集合...")
            try:
                reset_response = await client.post(f"{self.vector_service_url}/reset-collection")
                if reset_response.status_code == 200:
                    print("✅ Milvus集合重置成功")
                else:
                    print(f"⚠️  集合重置响应: {reset_response.status_code}")
            except Exception as e:
                print(f"⚠️  集合重置失败: {e}")
            
            # 等待集合重置完成
            await asyncio.sleep(3)
            
            # 2. 向量化测试文档
            print("🔢 向量化测试文档...")
            vectorize_response = await client.post(
                f"{self.vector_service_url}/vectorize",
                json={
                    "texts": test_documents,
                    "store_vectors": True,
                    "document_ids": [f"test-doc-{i}" for i in range(len(test_documents))]
                }
            )
            
            if vectorize_response.status_code != 200:
                print(f"❌ 向量化失败: {vectorize_response.status_code}")
                print(vectorize_response.text)
                return False
                
            vectorize_data = vectorize_response.json()
            print(f"✅ 成功向量化 {len(vectorize_data['vectors'])} 个文档")
            print(f"📐 向量维度: {len(vectorize_data['vectors'][0])}")
            print(f"💾 存储向量数: {vectorize_data['stored_count']}")
            
            # 等待存储完成
            await asyncio.sleep(5)
            
            # 3. 测试搜索匹配度
            print("\n🔍 测试搜索匹配度...")
            total_score = 0
            search_results = []
            
            for i, query in enumerate(test_queries):
                print(f"\n查询 {i+1}: {query}")
                
                start_time = time.time()
                search_response = await client.post(
                    f"{self.vector_service_url}/search",
                    json={
                        "query_text": query,
                        "top_k": 3
                    }
                )
                search_time = time.time() - start_time
                
                if search_response.status_code != 200:
                    print(f"❌ 搜索失败: {search_response.status_code}")
                    continue
                    
                search_data = search_response.json()
                results = search_data.get("results", [])
                
                print(f"⏱️  搜索耗时: {search_time:.3f}秒")
                print(f"📊 找到结果: {len(results)}个")
                
                if results:
                    best_match = results[0]
                    score = best_match.get("score", 0)
                    total_score += score
                    
                    print(f"🎯 最佳匹配 (相似度: {score:.3f}):")
                    print(f"   文本: {best_match.get('text', '')[:100]}...")
                    
                    search_results.append({
                        "query": query,
                        "best_score": score,
                        "search_time": search_time,
                        "result_count": len(results)
                    })
                else:
                    print("❌ 未找到匹配结果")
                    search_results.append({
                        "query": query,
                        "best_score": 0,
                        "search_time": search_time,
                        "result_count": 0
                    })
            
            # 4. 计算总体指标
            print(f"\n📈 优化效果评估:")
            avg_score = total_score / len(test_queries) if test_queries else 0
            avg_time = sum(r["search_time"] for r in search_results) / len(search_results)
            success_rate = sum(1 for r in search_results if r["best_score"] > 0) / len(search_results)
            
            print(f"✅ 平均相似度分数: {avg_score:.3f}")
            print(f"⚡ 平均搜索时间: {avg_time:.3f}秒") 
            print(f"🎯 成功匹配率: {success_rate:.1%}")
            
            # 评估改进效果
            if avg_score >= 0.8:
                print("🎉 优异! 搜索匹配度显著提升")
            elif avg_score >= 0.6:
                print("✅ 良好! 搜索质量有明显改善")
            elif avg_score >= 0.4:
                print("⚠️  一般! 仍有提升空间")
            else:
                print("❌ 较差! 需要进一步优化")
                
            return avg_score >= 0.6  # 60%以上认为优化成功
            
    async def test_chinese_support(self):
        """测试中文支持改进"""
        print("\n🇨🇳 测试中文处理能力...")
        
        chinese_docs = [
            "人工智能技术正在快速发展，深度学习算法取得了重大突破",
            "机器学习模型能够从大量数据中学习模式和规律",
            "自然语言处理使计算机能够理解和处理人类语言",
        ]
        
        chinese_queries = [
            "深度学习有什么突破？",
            "机器学习如何处理数据？",
            "NLP技术的应用"
        ]
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 向量化中文文档
            vectorize_response = await client.post(
                f"{self.vector_service_url}/vectorize", 
                json={
                    "texts": chinese_docs,
                    "store_vectors": True
                }
            )
            
            if vectorize_response.status_code != 200:
                print(f"❌ 中文向量化失败: {vectorize_response.status_code}")
                return False
                
            await asyncio.sleep(3)
            
            # 测试中文查询
            chinese_scores = []
            for query in chinese_queries:
                search_response = await client.post(
                    f"{self.vector_service_url}/search",
                    json={"query_text": query, "top_k": 2}
                )
                
                if search_response.status_code == 200:
                    results = search_response.json().get("results", [])
                    if results:
                        score = results[0].get("score", 0)
                        chinese_scores.append(score)
                        print(f"查询: {query} -> 匹配度: {score:.3f}")
            
            avg_chinese_score = sum(chinese_scores) / len(chinese_scores) if chinese_scores else 0
            print(f"🎯 中文查询平均匹配度: {avg_chinese_score:.3f}")
            
            if avg_chinese_score >= 0.7:
                print("✅ 中文处理能力优异")
                return True
            elif avg_chinese_score >= 0.5:
                print("⚠️  中文处理能力一般")
                return False
            else:
                print("❌ 中文处理能力需要改进")
                return False

async def main():
    """主测试函数"""
    print("🚀 TF-IDF向量化器优化测试开始...")
    print("=" * 60)
    
    tester = VectorOptimizationTester()
    
    try:
        # 检查服务健康状态
        async with httpx.AsyncClient() as client:
            health_response = await client.get("http://localhost:8002/health")
            if health_response.status_code != 200:
                print("❌ Vector Service不可用，请先启动服务")
                return
            
            health_data = health_response.json()
            print(f"✅ Vector Service状态: {health_data['status']}")
            print(f"🔧 Milvus连接: {health_data['components']['milvus']}")
        
        # 执行优化测试
        basic_success = await tester.test_vectorizer_config()
        chinese_success = await tester.test_chinese_support()
        
        print("\n" + "=" * 60)
        print("📊 最终测试结果:")
        
        if basic_success and chinese_success:
            print("🎉 优化成功! TF-IDF配置显著提升了搜索质量")
            print("✅ 基础搜索匹配度达标")
            print("✅ 中文处理能力优异")
        elif basic_success:
            print("✅ 部分成功! 基础搜索质量有改善，但中文处理仍需优化")
        else:
            print("❌ 优化效果不佳，需要进一步调整参数")
            
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())