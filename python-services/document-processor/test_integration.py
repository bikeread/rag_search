#!/usr/bin/env python3
"""
语义分块集成测试
验证新功能与现有文档处理系统的集成
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
import tempfile
import json

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

from src.document_processor import ModernDocumentProcessor
from langchain.schema import Document

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 测试文档内容
TEST_CONTENT = {
    "technical_report.md": """
# 机器学习项目性能分析

## 项目概览
本项目使用深度学习技术进行图像分类，在ImageNet数据集上达到了94.2%的准确率。

## 模型架构
- **基础模型**: ResNet-50
- **参数量**: 25.6M个参数
- **训练时间**: 48小时（使用8个V100 GPU）
- **推理速度**: 15.3ms/图像

## 性能指标
### 准确率表现
- Top-1准确率: 94.2%
- Top-5准确率: 99.1%
- 验证集损失: 0.087

### 计算资源使用
- GPU内存占用: 6.8GB
- CPU使用率: 45%
- 训练成本: $1,240

## 对比分析
与baseline模型相比：
- 准确率提升: +3.7%
- 推理速度提升: +28.5%
- 模型大小减少: -15.2%

结论：新模型在保持高准确率的同时，显著提升了推理效率。
""",

    "business_data.txt": """
季度财务报告

营收数据：
Q1营收: ¥8.2亿元 (+12.5%)
Q2营收: ¥9.7亿元 (+18.3%)
Q3营收: ¥11.4亿元 (+25.1%)
Q4营收: ¥13.8亿元 (+32.7%)

成本分析：
原材料成本占比: 42.3%
人工成本占比: 28.7%
运营成本占比: 15.8%
其他成本占比: 13.2%

利润指标：
毛利率: 57.7%
净利率: 23.4%
ROE: 18.9%
ROA: 12.6%

市场表现：
股价涨幅: +45.6%
市值: ¥458亿元
P/E ratio: 24.5
Market cap/Revenue: 3.2

风险评估：
流动比率: 2.1
负债率: 34.5%
现金流: 正向
信用评级: A+
""",

    "mixed_content.html": """
<!DOCTYPE html>
<html>
<head>
    <title>AI Research Findings</title>
</head>
<body>
    <h1>2024年人工智能研究报告</h1>
    
    <h2>Research Overview</h2>
    <p>Our team conducted extensive research on 15 different AI models, analyzing their performance across 8 benchmark datasets.</p>
    
    <h2>主要发现</h2>
    <ul>
        <li>GPT-4模型在文本理解任务上准确率达到89.4%</li>
        <li>Claude-3在数学推理上表现最佳，得分92.1分</li>
        <li>Gemini在多模态任务上领先，综合评分87.8分</li>
    </ul>
    
    <h2>Technical Specifications</h2>
    <table>
        <tr><th>Model</th><th>Parameters</th><th>Training Cost</th></tr>
        <tr><td>GPT-4</td><td>1.76T</td><td>$100M</td></tr>
        <tr><td>Claude-3</td><td>1.37T</td><td>$85M</td></tr>
        <tr><td>Gemini</td><td>1.56T</td><td>$120M</td></tr>
    </table>
    
    <h2>未来展望</h2>
    <p>预计到2025年，AI模型的参数量将达到10T级别，训练成本可能超过$500M。</p>
</body>
</html>
"""
}

async def test_document_processing_integration():
    """测试文档处理集成"""
    logger.info("开始测试文档处理集成...")
    
    # 初始化文档处理器
    processor = ModernDocumentProcessor()
    
    results = []
    
    for filename, content in TEST_CONTENT.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"处理文件: {filename}")
        
        try:
            # 确定MIME类型
            if filename.endswith('.md'):
                mime_type = 'text/markdown'
            elif filename.endswith('.txt'):
                mime_type = 'text/plain'
            elif filename.endswith('.html'):
                mime_type = 'text/html'
            else:
                mime_type = 'text/plain'
            
            # 处理文档
            result = await processor.process_file(
                file_content=content.encode('utf-8'),
                filename=filename,
                mime_type=mime_type,
                document_id=f"test_{filename.replace('.', '_')}",
                enable_chunking=True,
                chunk_size=800,
                chunk_overlap=200
            )
            
            logger.info(f"处理状态: {result.status}")
            logger.info(f"生成分块数量: {len(result.chunks)}")
            
            if result.status == "completed":
                # 分析分块结果
                for i, chunk in enumerate(result.chunks):
                    logger.info(f"\n  分块 {i+1}:")
                    logger.info(f"    长度: {len(chunk.content)} 字符")
                    logger.info(f"    分块方法: {chunk.metadata.get('chunking_method', 'unknown')}")
                    logger.info(f"    语义连贯性: {chunk.metadata.get('semantic_coherence', 'N/A')}")
                    logger.info(f"    包含关键信息: {chunk.metadata.get('contains_critical_info', 'N/A')}")
                    logger.info(f"    内容预览: {chunk.content[:100]}...")
                
                # 保存结果用于后续分析
                results.append({
                    'filename': filename,
                    'mime_type': mime_type,
                    'chunk_count': len(result.chunks),
                    'total_length': sum(len(chunk.content) for chunk in result.chunks),
                    'processing_metadata': result.metadata,
                    'chunks': [
                        {
                            'id': chunk.chunk_id,
                            'length': len(chunk.content),
                            'metadata': chunk.metadata
                        }
                        for chunk in result.chunks
                    ]
                })
            else:
                logger.error(f"处理失败: {result.error_message}")
        
        except Exception as e:
            logger.error(f"处理文件 {filename} 时出错: {str(e)}")
    
    return results

async def test_quality_consistency():
    """测试质量一致性"""
    logger.info("\n开始测试质量一致性...")
    
    processor = ModernDocumentProcessor()
    
    # 使用相同内容测试多次，检查结果一致性
    test_content = TEST_CONTENT["technical_report.md"]
    
    results = []
    
    for run in range(3):
        logger.info(f"第 {run + 1} 次运行...")
        
        try:
            result = await processor.process_file(
                file_content=test_content.encode('utf-8'),
                filename=f"consistency_test_run_{run}.md",
                mime_type='text/markdown',
                document_id=f"consistency_test_{run}",
                enable_chunking=True,
                chunk_size=600,
                chunk_overlap=150
            )
            
            if result.status == "completed":
                run_data = {
                    'run': run,
                    'chunk_count': len(result.chunks),
                    'avg_chunk_length': sum(len(c.content) for c in result.chunks) / len(result.chunks),
                    'semantic_chunks': sum(1 for c in result.chunks 
                                         if c.metadata.get('chunking_method') == 'semantic_embedding'),
                    'quality_metrics': []
                }
                
                # 收集质量指标
                for chunk in result.chunks:
                    if 'semantic_coherence' in chunk.metadata:
                        run_data['quality_metrics'].append(chunk.metadata['semantic_coherence'])
                
                results.append(run_data)
                
                logger.info(f"  分块数量: {run_data['chunk_count']}")
                logger.info(f"  平均分块长度: {run_data['avg_chunk_length']:.1f}")
                logger.info(f"  语义分块数量: {run_data['semantic_chunks']}")
        
        except Exception as e:
            logger.error(f"第 {run + 1} 次运行失败: {str(e)}")
    
    # 分析一致性
    if len(results) >= 2:
        chunk_counts = [r['chunk_count'] for r in results]
        avg_lengths = [r['avg_chunk_length'] for r in results]
        
        logger.info(f"\n一致性分析:")
        logger.info(f"  分块数量变化: {min(chunk_counts)} - {max(chunk_counts)}")
        logger.info(f"  平均长度变化: {min(avg_lengths):.1f} - {max(avg_lengths):.1f}")
        
        # 计算变异系数
        import statistics
        if len(chunk_counts) > 1:
            cv_count = statistics.stdev(chunk_counts) / statistics.mean(chunk_counts)
            cv_length = statistics.stdev(avg_lengths) / statistics.mean(avg_lengths)
            
            logger.info(f"  分块数量CV: {cv_count:.3f}")
            logger.info(f"  平均长度CV: {cv_length:.3f}")
            
            if cv_count < 0.1 and cv_length < 0.1:
                logger.info("  ✅ 一致性测试通过")
            else:
                logger.warning("  ⚠️  一致性可能存在问题")

async def test_performance_characteristics():
    """测试性能特征"""
    logger.info("\n开始测试性能特征...")
    
    processor = ModernDocumentProcessor()
    
    import time
    
    # 测试不同大小的文档
    test_sizes = [
        ("小文档", TEST_CONTENT["business_data.txt"][:500]),
        ("中文档", TEST_CONTENT["technical_report.md"]),
        ("大文档", TEST_CONTENT["mixed_content.html"] + TEST_CONTENT["technical_report.md"] * 3)
    ]
    
    for size_name, content in test_sizes:
        logger.info(f"\n测试 {size_name} (长度: {len(content)} 字符)")
        
        try:
            start_time = time.time()
            
            result = await processor.process_file(
                file_content=content.encode('utf-8'),
                filename=f"perf_test_{size_name}.txt",
                mime_type='text/plain',
                document_id=f"perf_test_{size_name}",
                enable_chunking=True,
                chunk_size=500,
                chunk_overlap=100
            )
            
            processing_time = time.time() - start_time
            
            if result.status == "completed":
                logger.info(f"  处理时间: {processing_time:.3f}s")
                logger.info(f"  生成分块: {len(result.chunks)}个")
                logger.info(f"  处理速度: {len(content)/processing_time:.0f} 字符/秒")
                
                # 计算分块质量分布
                semantic_count = sum(1 for c in result.chunks 
                                   if c.metadata.get('chunking_method') == 'semantic_embedding')
                fallback_count = len(result.chunks) - semantic_count
                
                logger.info(f"  语义分块: {semantic_count}个")
                logger.info(f"  降级分块: {fallback_count}个")
                
                if semantic_count > 0:
                    coherence_scores = [
                        c.metadata.get('semantic_coherence', 0)
                        for c in result.chunks
                        if 'semantic_coherence' in c.metadata
                    ]
                    if coherence_scores:
                        avg_coherence = sum(coherence_scores) / len(coherence_scores)
                        logger.info(f"  平均连贯性: {avg_coherence:.3f}")
        
        except Exception as e:
            logger.error(f"性能测试失败 ({size_name}): {str(e)}")

async def test_error_handling():
    """测试错误处理"""
    logger.info("\n开始测试错误处理...")
    
    processor = ModernDocumentProcessor()
    
    # 测试各种错误情况
    error_cases = [
        ("空内容", "", "text/plain"),
        ("无效MIME类型", "test content", "invalid/mime"),
        ("超大文档", "x" * 100000, "text/plain"),  # 100KB文档
        ("特殊字符", "🚀💻📊🔥⭐", "text/plain"),
    ]
    
    for case_name, content, mime_type in error_cases:
        logger.info(f"\n测试错误案例: {case_name}")
        
        try:
            result = await processor.process_file(
                file_content=content.encode('utf-8'),
                filename=f"error_test_{case_name}.txt",
                mime_type=mime_type,
                document_id=f"error_test_{case_name}",
                enable_chunking=True,
                chunk_size=500,
                chunk_overlap=100
            )
            
            logger.info(f"  处理状态: {result.status}")
            
            if result.status == "completed":
                logger.info(f"  ✅ 成功处理，生成 {len(result.chunks)} 个分块")
            else:
                logger.info(f"  ⚠️  处理失败: {result.error_message}")
        
        except Exception as e:
            logger.info(f"  ❌ 异常处理: {str(e)}")

async def generate_integration_report(test_results):
    """生成集成测试报告"""
    logger.info("\n生成集成测试报告...")
    
    report = {
        "test_summary": {
            "total_files_tested": len(test_results),
            "successful_processes": sum(1 for r in test_results if r.get('chunk_count', 0) > 0),
            "total_chunks_generated": sum(r.get('chunk_count', 0) for r in test_results),
            "average_chunks_per_file": sum(r.get('chunk_count', 0) for r in test_results) / len(test_results) if test_results else 0
        },
        "file_analysis": test_results,
        "recommendations": []
    }
    
    # 生成建议
    if report["test_summary"]["successful_processes"] < len(test_results):
        report["recommendations"].append("部分文件处理失败，需要检查错误处理机制")
    
    if report["test_summary"]["average_chunks_per_file"] > 10:
        report["recommendations"].append("平均分块数量较高，考虑调整分块大小参数")
    
    # 保存报告
    report_file = Path("integration_test_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"集成测试报告已保存到: {report_file}")
    
    # 显示摘要
    logger.info(f"\n📊 集成测试摘要:")
    logger.info(f"  测试文件数: {report['test_summary']['total_files_tested']}")
    logger.info(f"  成功处理数: {report['test_summary']['successful_processes']}")
    logger.info(f"  总分块数: {report['test_summary']['total_chunks_generated']}")
    logger.info(f"  平均分块数: {report['test_summary']['average_chunks_per_file']:.1f}")
    
    if report["recommendations"]:
        logger.info(f"\n💡 建议:")
        for rec in report["recommendations"]:
            logger.info(f"  - {rec}")

async def main():
    """主测试函数"""
    logger.info("开始语义分块集成测试")
    
    try:
        # 文档处理集成测试
        test_results = await test_document_processing_integration()
        
        # 质量一致性测试
        await test_quality_consistency()
        
        # 性能特征测试
        await test_performance_characteristics()
        
        # 错误处理测试
        await test_error_handling()
        
        # 生成测试报告
        await generate_integration_report(test_results)
        
        logger.info("\n✅ 所有集成测试完成")
        
    except Exception as e:
        logger.error(f"❌ 集成测试过程中出现错误: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    # 运行集成测试
    exit_code = asyncio.run(main())
    sys.exit(exit_code)