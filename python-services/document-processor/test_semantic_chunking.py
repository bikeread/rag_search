#!/usr/bin/env python3
"""
语义分块功能测试脚本
测试新实现的语义感知分块和质量评估功能
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "src"))

from src.splitting.semantic_chunker import EmbeddingBasedSemanticChunker
from src.splitting.chunk_quality_assessor import ChunkQualityAssessor, ChunkQualityReporter
from src.config.semantic_chunking_config import get_config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 测试文档样本
TEST_DOCUMENTS = {
    "technical_doc": """
# GitHub开源项目分析报告

## 项目统计数据

最受欢迎的JavaScript项目包括：

1. **React** - Facebook开发的前端框架
   - Star数量: 223,000⭐
   - Fork数量: 45,600
   - 贡献者: 1,800+人
   - 创建时间: 2013年5月29日

2. **Vue.js** - 渐进式前端框架  
   - Star数量: 206,000⭐
   - Fork数量: 33,400
   - 贡献者: 440+人
   - 创建时间: 2014年2月14日

3. **Node.js** - JavaScript运行时环境
   - Star数量: 102,000⭐
   - Fork数量: 28,100
   - 贡献者: 3,200+人
   - 创建时间: 2009年5月27日

## 技术特性分析

React采用虚拟DOM技术，性能提升达到40-60%。Vue.js的数据绑定机制使开发效率提升35%。Node.js的事件循环架构支持高并发，处理能力可达10,000个并发连接。

### 市场占有率

根据2024年开发者调研：
- React: 68.9%的开发者在使用
- Vue.js: 18.8%的开发者在使用  
- Angular: 17.3%的开发者在使用
- Node.js: 47.2%的后端开发者选择

这些数据表明React在前端领域占据主导地位，而Node.js在后端JavaScript开发中也有重要地位。
""",

    "business_doc": """
公司年度财务报告摘要

营收情况：
2024年总营收达到¥15.8亿元，同比增长23.5%。其中：
- 产品销售收入：¥12.3亿元（占比77.8%）
- 服务收入：¥2.8亿元（占比17.7%） 
- 其他收入：¥0.7亿元（占比4.4%）

成本结构：
总成本¥9.2亿元，毛利率达到41.8%。主要成本构成：
- 原材料成本：¥5.1亿元
- 人工成本：¥2.8亿元
- 运营成本：¥1.3亿元

利润指标：
净利润¥3.2亿元，净利率20.3%。每股收益¥1.85元，同比增长18.9%。

现金流量：
经营活动现金流¥4.7亿元，投资活动现金流-¥1.8亿元，筹资活动现金流¥0.3亿元。
期末现金及现金等价物余额¥8.9亿元。

未来展望：
预计2025年营收将达到¥19-21亿元，增长率保持在20-25%区间。
""",

    "mixed_language_doc": """
AI大模型技术发展报告

## Introduction
Artificial Intelligence has transformed the technology landscape in 2024. Large Language Models (LLMs) like GPT-4, Claude, and Gemini have achieved remarkable capabilities.

## 市场现状
当前AI大模型市场规模约为$50.2 billion，预计到2025年将达到$78.4 billion，增长率为56.1%。

主要厂商情况：
- OpenAI GPT-4: 1,750亿参数，训练成本约$100 million
- Google Gemini: 1,560亿参数，多模态能力领先
- Anthropic Claude: 1,370亿参数，安全性评分最高

## Technical Specifications
模型性能评估结果：
- MMLU benchmark: GPT-4 (86.4%), Gemini (90.0%), Claude (88.7%)
- HumanEval coding: GPT-4 (82.1%), Gemini (78.9%), Claude (85.3%)
- 中文理解测试: GPT-4 (78.2%), Gemini (81.5%), Claude (83.1%)

计算资源需求：
Training requires approximately 10,000 A100 GPUs for 2-3 months.
推理成本每1000 tokens约$0.03-0.06。

The future of AI development will focus on efficiency and specialization.
""",

    "simple_text": """
这是一个简单的测试文档。它包含几个句子来测试基本的分块功能。

第一段介绍了测试的目的。我们想要验证语义分块器是否能够正确处理简单的中文文本。

第二段讨论了一些具体的数字。比如这个测试有3个段落，总共大约200个字符。

最后一段总结了测试内容。我们期望分块器能够保持段落的完整性，同时正确识别关键信息。
"""
}

async def test_semantic_chunker():
    """测试语义分块器"""
    logger.info("开始测试语义分块器...")
    
    # 初始化分块器
    chunker = EmbeddingBasedSemanticChunker(
        vector_service_url="http://localhost:8002",
        coherence_threshold=0.7,
        min_chunk_size=200,
        max_chunk_size=1000,
        critical_info_protection=True
    )
    
    for doc_name, doc_text in TEST_DOCUMENTS.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"测试文档: {doc_name}")
        logger.info(f"文档长度: {len(doc_text)} 字符")
        
        try:
            # 执行语义分块
            chunks = await chunker.chunk_document(
                text=doc_text,
                metadata={"document_name": doc_name, "test_run": True}
            )
            
            logger.info(f"生成分块数量: {len(chunks)}")
            
            # 显示分块结果
            for i, chunk in enumerate(chunks):
                logger.info(f"\n--- Chunk {i+1} ---")
                logger.info(f"长度: {len(chunk.text)} 字符")
                logger.info(f"语义连贯性: {chunk.semantic_coherence_score:.3f}")
                logger.info(f"包含关键信息: {chunk.contains_critical_info}")
                logger.info(f"起始位置: {chunk.start_position}")
                logger.info(f"结束位置: {chunk.end_position}")
                logger.info(f"内容预览: {chunk.text[:100]}...")
                
                # 显示元数据
                metadata = chunk.metadata
                if metadata:
                    logger.info(f"元数据: {metadata}")
            
        except Exception as e:
            logger.error(f"测试文档 {doc_name} 时出错: {str(e)}")

async def test_quality_assessor():
    """测试质量评估器"""
    logger.info("\n开始测试质量评估器...")
    
    # 初始化评估器
    assessor = ChunkQualityAssessor(vector_service_url="http://localhost:8002")
    reporter = ChunkQualityReporter()
    
    # 使用技术文档进行测试
    doc_text = TEST_DOCUMENTS["technical_doc"]
    
    # 创建两种不同的分块方式进行对比
    test_chunks = [
        # 模拟语义分块结果
        {
            "text": doc_text[:500],
            "metadata": {"chunking_method": "semantic", "contains_numbers": True}
        },
        {
            "text": doc_text[500:1000],
            "metadata": {"chunking_method": "semantic", "contains_numbers": True}
        },
        {
            "text": doc_text[1000:],
            "metadata": {"chunking_method": "semantic", "contains_numbers": False}
        }
    ]
    
    try:
        # 执行质量评估
        quality_metrics = await assessor.assess_chunking_quality(
            original_text=doc_text,
            chunks=test_chunks,
            chunk_method="semantic_test"
        )
        
        # 生成质量报告
        quality_report = reporter.generate_quality_report(quality_metrics)
        
        # 显示评估结果
        logger.info(f"\n质量评估结果:")
        logger.info(f"总体评分: {quality_metrics.overall_score:.3f}")
        logger.info(f"语义连贯性: {quality_metrics.semantic_coherence:.3f}")
        logger.info(f"信息保留度: {quality_metrics.information_preservation:.3f}")
        logger.info(f"大小一致性: {quality_metrics.size_consistency:.3f}")
        logger.info(f"边界质量: {quality_metrics.boundary_quality:.3f}")
        logger.info(f"检索有效性: {quality_metrics.retrieval_effectiveness:.3f}")
        
        logger.info(f"\n质量报告:")
        logger.info(f"质量等级: {quality_report['overall_assessment']['level']}")
        logger.info(f"评估摘要: {quality_report['overall_assessment']['summary']}")
        
        if quality_report['recommendations']:
            logger.info(f"\n改进建议:")
            for rec in quality_report['recommendations']:
                logger.info(f"- {rec}")
        
        if quality_report['improvement_priority']:
            logger.info(f"\n改进优先级: {quality_report['improvement_priority']}")
    
    except Exception as e:
        logger.error(f"质量评估测试失败: {str(e)}")

async def test_config_system():
    """测试配置系统"""
    logger.info("\n开始测试配置系统...")
    
    # 测试不同配置模板
    templates = ['default', 'high_quality', 'chinese_optimized', 'technical_documents']
    
    for template_name in templates:
        try:
            config = get_config(template_name)
            logger.info(f"\n配置模板: {template_name}")
            logger.info(f"  启用语义分块: {config.enable_semantic_chunking}")
            logger.info(f"  最小分块大小: {config.min_chunk_size}")
            logger.info(f"  最大分块大小: {config.max_chunk_size}")
            logger.info(f"  连贯性阈值: {config.coherence_threshold}")
            logger.info(f"  关键信息保护: {config.critical_info_protection}")
            
        except Exception as e:
            logger.error(f"测试配置模板 {template_name} 失败: {str(e)}")
    
    # 测试自定义配置
    try:
        custom_config = get_config('default', {
            'min_chunk_size': 150,
            'max_chunk_size': 800,
            'coherence_threshold': 0.8
        })
        logger.info(f"\n自定义配置:")
        logger.info(f"  最小分块大小: {custom_config.min_chunk_size}")
        logger.info(f"  最大分块大小: {custom_config.max_chunk_size}")
        logger.info(f"  连贯性阈值: {custom_config.coherence_threshold}")
        
    except Exception as e:
        logger.error(f"测试自定义配置失败: {str(e)}")

async def benchmark_chunking_methods():
    """基准测试不同分块方法"""
    logger.info("\n开始基准测试...")
    
    import time
    from src.splitting.strategies import SplitStrategy
    
    test_text = TEST_DOCUMENTS["technical_doc"]
    
    # 测试传统分块方法
    start_time = time.time()
    traditional_chunks = SplitStrategy.by_character(test_text, 1000, 200)
    traditional_time = time.time() - start_time
    
    logger.info(f"传统字符分块:")
    logger.info(f"  分块数量: {len(traditional_chunks)}")
    logger.info(f"  处理时间: {traditional_time:.3f}s")
    
    # 测试智能分块方法
    start_time = time.time()
    smart_chunks = SplitStrategy.semantic_aware_split(test_text, 1000, 200)
    smart_time = time.time() - start_time
    
    logger.info(f"智能语义分块:")
    logger.info(f"  分块数量: {len(smart_chunks)}")
    logger.info(f"  处理时间: {smart_time:.3f}s")
    
    # 测试嵌入式语义分块（如果向量服务可用）
    try:
        chunker = EmbeddingBasedSemanticChunker()
        start_time = time.time()
        semantic_chunks = await chunker.chunk_document(test_text)
        semantic_time = time.time() - start_time
        
        logger.info(f"嵌入式语义分块:")
        logger.info(f"  分块数量: {len(semantic_chunks)}")
        logger.info(f"  处理时间: {semantic_time:.3f}s")
        
    except Exception as e:
        logger.warning(f"嵌入式语义分块测试失败: {str(e)}")

async def main():
    """主测试函数"""
    logger.info("开始语义分块功能全面测试")
    
    try:
        # 测试配置系统
        await test_config_system()
        
        # 基准测试
        await benchmark_chunking_methods()
        
        # 测试语义分块器
        await test_semantic_chunker()
        
        # 测试质量评估器
        await test_quality_assessor()
        
        logger.info("\n✅ 所有测试完成")
        
    except Exception as e:
        logger.error(f"❌ 测试过程中出现错误: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    # 运行测试
    exit_code = asyncio.run(main())
    sys.exit(exit_code)