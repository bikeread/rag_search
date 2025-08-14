#!/usr/bin/env python3
"""
混合检索系统数值查询优化测试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import HybridRetriever, BM25TextRetriever, VectorServiceClient

def test_numerical_query_classification():
    """测试数值查询分类功能"""
    print("🧪 测试数值查询分类功能")
    
    # 模拟一个简单的向量客户端
    class MockVectorClient:
        async def search_vectors(self, query, top_k):
            return []
    
    vector_client = MockVectorClient()
    retriever = HybridRetriever(vector_client)
    
    # 测试用例
    test_cases = [
        ("哪个项目的star数量最多？", True, "应该识别为数值查询"),
        ("这三个有star的项目分别用什么编程语言开发？", False, "应该识别为技术查询"),
        ("dify_wechat_plugin有多少个star？", True, "应该识别为数值查询"),
        ("项目创建时间是什么时候？", True, "应该识别为时间相关数值查询"),
        ("bikeread有几个GitHub项目？", True, "应该识别为数量查询"),
        ("41个star的项目是哪个？", True, "应该识别为包含具体数字的查询"),
        ("最多star的项目排名第一", True, "应该识别为排名相关数值查询"),
        ("rag_search系统包含哪些主要组件？", False, "应该识别为组件分析查询"),
    ]
    
    passed = 0
    failed = 0
    
    for query, expected, description in test_cases:
        result = retriever._is_numerical_query(query)
        status = "✅" if result == expected else "❌"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
            
        print(f"{status} {query} -> {result} ({description})")
    
    print(f"\n📊 测试结果: {passed}通过, {failed}失败")
    return failed == 0

def test_dynamic_weights():
    """测试动态权重分配"""
    print("\n🧪 测试动态权重分配")
    
    class MockVectorClient:
        async def search_vectors(self, query, top_k):
            return []
    
    vector_client = MockVectorClient()
    retriever = HybridRetriever(vector_client)
    
    test_cases = [
        ("哪个项目star最多？", (0.2, 0.8), "精确数值比较查询"),
        ("41个star", (0.25, 0.75), "包含具体数字的查询"),
        ("多少个star？", (0.3, 0.7), "一般数值查询"),
        ("项目架构设计", (0.75, 0.25), "架构分析查询"),
        ("用什么编程语言？", (0.7, 0.3), "编程语言查询"),
        ("A项目和B项目的区别", (0.5, 0.5), "对比查询"),
        ("一般查询", (0.6, 0.4), "默认权重"),
    ]
    
    passed = 0
    failed = 0
    
    for query, expected_weights, description in test_cases:
        vector_weight, bm25_weight = retriever._get_dynamic_weights(query)
        actual_weights = (vector_weight, bm25_weight)
        
        # 允许小的浮点数误差
        weight_match = (abs(actual_weights[0] - expected_weights[0]) < 0.01 and 
                       abs(actual_weights[1] - expected_weights[1]) < 0.01)
        
        status = "✅" if weight_match else "❌"
        
        if weight_match:
            passed += 1
        else:
            failed += 1
            
        print(f"{status} {query} -> vector:{vector_weight:.2f}, bm25:{bm25_weight:.2f} ({description})")
    
    print(f"\n📊 测试结果: {passed}通过, {failed}失败")
    return failed == 0

def test_numerical_info_extraction():
    """测试数值信息提取"""
    print("\n🧪 测试数值信息提取")
    
    bm25_retriever = BM25TextRetriever()
    
    test_text = """
    dify_wechat_plugin项目有41个star，是star数量最多的项目。
    rag_search项目有8个star和1个fork。
    thesis_work_flow项目有4个star，创建于2023年。
    总共有3个项目获得了star。
    """
    
    extracted = bm25_retriever.extract_numerical_info(test_text)
    
    print("提取到的数值信息:")
    for key, values in extracted.items():
        print(f"  {key}: {values}")
    
    # 验证关键数值是否被提取
    expected_stars = ['41', '8', '4']
    expected_years = ['2023']
    expected_projects = ['3']
    
    stars_found = extracted.get('stars', [])
    years_found = extracted.get('years', [])
    projects_found = extracted.get('projects_count', [])
    
    stars_match = all(star in stars_found for star in expected_stars)
    years_match = all(year in years_found for year in expected_years)
    projects_match = any(proj in projects_found for proj in expected_projects)
    
    print(f"\n验证结果:")
    print(f"✅ Stars提取: {stars_match} - 期望{expected_stars}, 实际{stars_found}")
    print(f"✅ Years提取: {years_match} - 期望{expected_years}, 实际{years_found}")
    print(f"✅ Projects提取: {projects_match} - 期望{expected_projects}, 实际{projects_found}")
    
    return stars_match and years_match and projects_match

def test_text_preprocessing():
    """测试文本预处理优化"""
    print("\n🧪 测试文本预处理优化")
    
    bm25_retriever = BM25TextRetriever()
    
    test_text = "dify_wechat_plugin有41个star，这是最多的star数量"
    
    tokens = bm25_retriever._preprocess_text_v2(test_text)
    
    print(f"原文: {test_text}")
    print(f"分词结果: {tokens}")
    
    # 验证关键token是否被正确提取
    expected_tokens = ['dify_wechat_plugin', '41', 'star', 'stars_41', '41_stars', '最多', '数量']
    found_important_tokens = [token for token in expected_tokens if token in tokens]
    
    print(f"重要token提取: {found_important_tokens}")
    
    # 至少应该提取到项目名、数字、star等关键信息
    has_project = 'dify_wechat_plugin' in tokens
    has_number = '41' in tokens  
    has_star = 'star' in tokens or 'stars' in tokens
    
    success = has_project and has_number and has_star
    status = "✅" if success else "❌"
    
    print(f"{status} 关键信息提取: 项目名={has_project}, 数字={has_number}, star={has_star}")
    
    return success

def main():
    """运行所有测试"""
    print("🚀 混合检索系统数值查询优化测试")
    print("=" * 50)
    
    results = []
    
    # 运行各项测试
    results.append(test_numerical_query_classification())
    results.append(test_dynamic_weights())
    results.append(test_numerical_info_extraction())
    results.append(test_text_preprocessing())
    
    print("\n" + "=" * 50)
    print("📊 总体测试结果")
    
    passed_tests = sum(results)
    total_tests = len(results)
    
    print(f"✅ 通过测试: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！数值查询优化功能正常")
        return 0
    else:
        print("⚠️  部分测试失败，需要进一步调试")
        return 1

if __name__ == "__main__":
    exit(main())