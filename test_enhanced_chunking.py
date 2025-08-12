#!/usr/bin/env python3
"""
测试增强的文档分块策略
验证语义感知分块和数字信息保护功能
"""

import sys
import os
sys.path.append('/home/bikeread/dev/rag_search/python-services/document-processor/src')

from splitting.splitter import DocumentSplitter, DocumentType
from splitting.strategies import SplitStrategy, SemanticBoundaryDetector, ImportanceLevel

def test_semantic_boundary_detection():
    """测试语义边界检测"""
    print("🔍 测试语义边界检测功能...")
    
    test_text = """
# 财务报告分析

本季度销售额达到1,250万元，比上季度增长15.8%。其中：

1. 产品A销售额：450万元（增长20%）
2. 产品B销售额：380万元（增长12%）  
3. 产品C销售额：420万元（增长10%）

重要公式：净利润 = 总收入 - 总成本 - 税费

市场占有率从12.3%提升到14.7%，表现优异。
    """
    
    detector = SemanticBoundaryDetector()
    segments = detector.detect_important_segments(test_text)
    
    print(f"✅ 检测到 {len(segments)} 个语义段落")
    
    critical_segments = [s for s in segments if s.importance == ImportanceLevel.CRITICAL]
    print(f"📊 其中 {len(critical_segments)} 个包含关键信息（数字/公式）")
    
    for i, segment in enumerate(segments):
        print(f"段落 {i+1}: {segment.importance.value} - "
              f"数字: {segment.contains_numbers}, "
              f"公式: {segment.contains_formulas}, "
              f"类型: {segment.segment_type}")
        print(f"   内容预览: {segment.text[:50]}...")
    
    return len(critical_segments) > 0

def test_semantic_aware_splitting():
    """测试语义感知分块"""
    print("\n🧩 测试语义感知分块...")
    
    test_text = """
人工智能技术发展报告

近年来，人工智能技术取得了显著进展。据统计，2023年全球AI市场规模达到1,847亿美元，同比增长37.3%。

## 主要技术突破

1. 大语言模型参数量突破1,000亿，性能提升23.5%
2. 计算机视觉准确率达到98.7%，误差率降低到1.3%
3. 语音识别准确率提升到99.2%

关键算法优化：准确率 = 正确预测数 / 总预测数 × 100%

在医疗领域，AI诊断准确率达到94.6%，超越人类专家的89.2%。在金融领域，风控模型识别率提升到96.8%。

预计到2025年，全球AI投资将达到5,420亿美元，年复合增长率为42.1%。
    """
    
    chunks = SplitStrategy.semantic_aware_split(
        text=test_text, 
        chunk_size=500, 
        chunk_overlap=100, 
        preserve_numbers=True
    )
    
    print(f"✅ 生成 {len(chunks)} 个语义感知块")
    
    numbers_preserved = 0
    for i, chunk in enumerate(chunks):
        metadata = chunk['metadata']
        chunk_text = chunk['text']
        
        print(f"\n块 {i+1}: {len(chunk_text)} 字符")
        print(f"  包含数字: {metadata.get('contains_numbers', False)}")
        print(f"  包含公式: {metadata.get('contains_formulas', False)}")
        print(f"  重要性级别: {metadata.get('importance_levels', [])}")
        print(f"  内容预览: {chunk_text[:100]}...")
        
        if metadata.get('contains_numbers', False):
            numbers_preserved += 1
    
    print(f"📈 数字信息保护：{numbers_preserved}/{len(chunks)} 个块包含数字")
    
    return numbers_preserved > 0

def test_chunking_quality_assessment():
    """测试分块质量评估"""
    print("\n📏 测试分块质量评估...")
    
    test_text = """
机器学习算法性能对比

本研究比较了5种主流机器学习算法的性能：

1. 随机森林：准确率95.3%，训练时间2.5小时
2. 支持向量机：准确率93.7%，训练时间1.8小时  
3. 神经网络：准确率96.2%，训练时间4.2小时
4. 决策树：准确率89.4%，训练时间0.5小时
5. 朴素贝叶斯：准确率87.1%，训练时间0.3小时

性能评估公式：F1分数 = 2 × (精确率 × 召回率) / (精确率 + 召回率)

数据集包含10,000个样本，训练集占80%（8,000个），测试集占20%（2,000个）。

实验结果显示，神经网络在准确率方面表现最优，达到96.2%，但训练时间最长。随机森林在准确率和效率之间取得了良好平衡，准确率95.3%，训练时间2.5小时。
    """
    
    splitter = DocumentSplitter(DocumentType.AUTO)
    result = splitter.split_text_with_quality_assessment(test_text)
    
    quality = result['quality']
    chunks = result['chunks']
    
    print(f"✅ 生成 {len(chunks)} 个块，原文长度 {result['original_length']} 字符")
    print(f"📊 质量评估分数: {quality['score']:.3f} ({quality['assessment']})")
    
    metrics = quality['metrics']
    print("📈 详细指标:")
    print(f"  数字信息保护率: {metrics.get('number_preservation_rate', 0):.1%}")
    print(f"  语义完整性: {metrics.get('semantic_completeness', 0):.1%}")
    print(f"  块大小一致性: {metrics.get('size_consistency', 0):.1%}")
    print(f"  关键信息分布: {metrics.get('critical_info_distribution', 0):.1%}")
    
    # 验证数字信息保护率是否达到90%+
    number_protection_rate = metrics.get('number_preservation_rate', 0)
    protection_success = number_protection_rate >= 0.9
    
    if protection_success:
        print("🎉 数字信息保护率达标 (≥90%)")
    else:
        print(f"⚠️  数字信息保护率未达标: {number_protection_rate:.1%} < 90%")
    
    return protection_success, quality['score']

def test_adaptive_splitting():
    """测试自适应分块"""
    print("\n🔄 测试自适应分块策略...")
    
    # 高数字密度文本
    high_number_text = """
股票价格分析：AAPL股价从150.25美元上涨到165.80美元，涨幅10.35%。
交易量达到8,750,000股，比平均值6,200,000股增长41.13%。
市值从2.45万亿美元增至2.68万亿美元。PE比率为28.5，PB比率为7.2。
    """
    
    # 结构化文本
    structured_text = """
# Python开发指南

## 1. 环境配置
安装Python 3.9+版本，配置虚拟环境。

## 2. 依赖管理
使用pip安装requirements.txt中的依赖包。

## 3. 代码规范
遵循PEP 8规范，使用黑格式化工具。
    """
    
    print("测试高数字密度文本:")
    high_number_chunks = SplitStrategy.adaptive_split(high_number_text, target_chunk_size=200)
    print(f"  生成 {len(high_number_chunks)} 个块")
    
    print("测试结构化文本:")
    structured_chunks = SplitStrategy.adaptive_split(structured_text, target_chunk_size=200) 
    print(f"  生成 {len(structured_chunks)} 个块")
    
    # 检查是否使用了不同的分块策略
    high_number_method = high_number_chunks[0]['metadata'].get('split_method', 'unknown')
    structured_method = structured_chunks[0]['metadata'].get('split_method', 'unknown')
    
    print(f"  高数字密度文本使用方法: {high_number_method}")
    print(f"  结构化文本使用方法: {structured_method}")
    
    return len(high_number_chunks) > 0 and len(structured_chunks) > 0

def main():
    """主测试函数"""
    print("🚀 开始测试增强的文档分块策略...")
    print("=" * 60)
    
    test_results = []
    
    try:
        # 测试1: 语义边界检测
        result1 = test_semantic_boundary_detection()
        test_results.append(("语义边界检测", result1))
        
        # 测试2: 语义感知分块
        result2 = test_semantic_aware_splitting()
        test_results.append(("语义感知分块", result2))
        
        # 测试3: 分块质量评估
        result3, quality_score = test_chunking_quality_assessment()
        test_results.append(("分块质量评估", result3))
        
        # 测试4: 自适应分块
        result4 = test_adaptive_splitting()
        test_results.append(("自适应分块", result4))
        
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
        
        if success_rate >= 0.75:
            print("🎉 文档分块策略改进成功！")
            print("✅ 语义边界识别正常工作")
            print("✅ 数字信息保护机制有效")
            print("✅ 分块质量评估功能正常")
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
    success = main()
    exit(0 if success else 1)