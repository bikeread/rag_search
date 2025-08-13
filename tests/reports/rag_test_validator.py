#!/usr/bin/env python3
"""
RAG System Comprehensive Validation Test Suite
专业RAG系统30问题测试套件 - 混合检索BM25+向量搜索验证
"""

import httpx
import json
import time
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import asyncio

class RAGSystemValidator:
    def __init__(self):
        self.base_url = "http://localhost:3001"
        self.jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbWU1OHhpbnUwMDAwa2NhMWZjdGxpaXNqIiwiZW1haWwiOiJ0ZXN0QGV4YW1wbGUuY29tIiwibmFtZSI6IlRlc3QgVXNlciIsInJvbGUiOiJVU0VSIiwiaWF0IjoxNzU0OTk4MjQwLCJleHAiOjE3NTQ5OTkxNDB9.d6OyMRi6YohB6EKx_XC_6h0_5wvuJWR9JJhQX57M274"
        self.document_id = "cme8gn7nn002arc9k88c3uk46"
        self.headers = {
            "Authorization": f"Bearer {self.jwt_token}",
            "Content-Type": "application/json"
        }
        
        # 30个测试问题分为3个阶段
        self.test_questions = {
            "Phase 1: Basic Information Retrieval": [
                "bikeread有哪些GitHub项目获得了star？",
                "这三个有star的项目分别用什么编程语言开发？", 
                "哪个项目的star数量最多？",
                "rag_search项目什么时候创建的，项目规模多大？",
                "thesis_work_flow是最新的项目吗？",
                "dify_wechat_plugin的核心功能和应用场景是什么？",
                "rag_search系统包含哪些主要组件？",
                "thesis_work_flow是做什么的？",
                "微信公众号插件如何处理多种消息类型？",
                "RAG系统使用了哪些数据库？"
            ],
            "Phase 2: Technical Analysis": [
                "bikeread的三个项目分别采用了什么技术架构？",
                "rag_search如何实现向量搜索功能？",
                "dify_wechat_plugin如何解决微信15秒响应限制？",
                "哪些项目使用了Python语言？",
                "rag_search的微服务架构是怎样的？",
                "比较三个项目的技术栈和应用领域",
                "哪个项目最适合企业应用？",
                "bikeread的项目体现了哪些技术发展趋势？",
                "从创建时间看，bikeread的项目发展轨迹如何？",
                "三个项目中哪个社区影响力最大？"
            ],
            "Phase 3: Advanced Analysis": [
                "rag_search系统如何实现端到端的文档查询？",
                "dify_wechat_plugin的配置参数有哪些类型？",
                "thesis_work_flow在学术研究中的应用价值是什么？",
                "这些项目在AI技术应用方面有什么特色？",
                "rag_search使用的Ollama LLM有什么特点？",
                "如何选择合适的项目进行AI应用开发？",
                "部署rag_search系统需要哪些环境准备？",
                "使用dify_wechat_plugin需要注意哪些配置要点？",
                "开发类似的AI集成项目应该考虑什么？",
                "bikeread的开发经验对其他开发者有什么启发？"
            ]
        }
        
        self.results = []
        
    async def send_query(self, question: str) -> Dict:
        """发送查询到RAG系统并记录结果"""
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {
                    "query": question,
                    "retrieval_mode": "hybrid",
                    "vector_weight": 0.8,
                    "bm25_weight": 0.2,
                    "top_k": 3
                }
                
                response = await client.post(
                    f"{self.base_url}/api/query",
                    json=payload,
                    headers=self.headers
                )
                
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "response_time": response_time,
                        "answer": result.get("answer", ""),
                        "sources": result.get("sources", []),
                        "metadata": result.get("metadata", {}),
                        "raw_response": result
                    }
                else:
                    return {
                        "success": False,
                        "response_time": response_time,
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "answer": "",
                        "sources": [],
                        "metadata": {}
                    }
                    
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "success": False,
                "response_time": response_time,
                "error": str(e),
                "answer": "",
                "sources": [],
                "metadata": {}
            }
    
    def evaluate_answer(self, question: str, answer: str, sources: List) -> Dict:
        """评估回答质量"""
        # 基础评分标准
        accuracy_score = 0
        completeness_score = 0
        relevance_score = 0
        
        answer_lower = answer.lower()
        question_lower = question.lower()
        
        # 根据不同类型的问题进行评分
        if "star" in question_lower:
            if any(keyword in answer_lower for keyword in ["star", "标星", "收藏"]):
                accuracy_score = 90
                if any(num in answer for num in ["3", "2", "1"]):
                    completeness_score = 95
            else:
                accuracy_score = 20
                
        elif "编程语言" in question_lower or "language" in question_lower:
            languages = ["python", "typescript", "javascript", "node.js"]
            found_languages = [lang for lang in languages if lang in answer_lower]
            accuracy_score = min(90, len(found_languages) * 30)
            completeness_score = min(95, len(found_languages) * 25)
            
        elif "项目" in question_lower and ("创建" in question_lower or "时间" in question_lower):
            if any(keyword in answer_lower for keyword in ["2024", "2023", "创建", "时间"]):
                accuracy_score = 85
                completeness_score = 80
                
        elif "rag" in question_lower:
            rag_keywords = ["向量", "搜索", "数据库", "milvus", "postgresql", "fastapi"]
            found_keywords = [kw for kw in rag_keywords if kw in answer_lower]
            accuracy_score = min(90, len(found_keywords) * 15)
            completeness_score = min(90, len(found_keywords) * 15)
            
        elif "微信" in question_lower:
            wechat_keywords = ["微信", "公众号", "消息", "响应", "15秒"]
            found_keywords = [kw for kw in wechat_keywords if kw in answer_lower]
            accuracy_score = min(90, len(found_keywords) * 18)
            completeness_score = min(85, len(found_keywords) * 17)
            
        else:
            # 通用评分：基于答案长度和来源数量
            if len(answer) > 50:
                accuracy_score = 70
            if len(sources) > 0:
                accuracy_score += 15
            if len(answer) > 100:
                completeness_score = 75
            if "详细" in answer_lower or "具体" in answer_lower:
                completeness_score += 10
        
        # 相关性评分：基于来源数量和答案内容
        relevance_score = min(90, len(sources) * 30)
        if len(answer) > 20:
            relevance_score += 10
            
        # 确保分数在合理范围内
        accuracy_score = max(0, min(100, accuracy_score))
        completeness_score = max(0, min(100, completeness_score))
        relevance_score = max(0, min(100, relevance_score))
        
        return {
            "accuracy": accuracy_score,
            "completeness": completeness_score,
            "relevance": relevance_score,
            "overall": (accuracy_score + completeness_score + relevance_score) / 3
        }
    
    async def run_test_phase(self, phase_name: str, questions: List[str]) -> List[Dict]:
        """运行单个测试阶段"""
        print(f"\n{'='*60}")
        print(f"开始执行: {phase_name}")
        print(f"{'='*60}")
        
        phase_results = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n问题 {i}/{len(questions)}: {question}")
            print("-" * 50)
            
            # 发送查询
            result = await self.send_query(question)
            
            # 评估答案
            if result["success"]:
                evaluation = self.evaluate_answer(question, result["answer"], result["sources"])
                print(f"✅ 响应时间: {result['response_time']:.2f}s")
                print(f"📊 评分 - 准确性: {evaluation['accuracy']:.1f}% | 完整性: {evaluation['completeness']:.1f}% | 相关性: {evaluation['relevance']:.1f}%")
                print(f"🎯 综合得分: {evaluation['overall']:.1f}%")
                print(f"📄 来源数量: {len(result['sources'])}")
                print(f"💬 答案预览: {result['answer'][:100]}...")
            else:
                evaluation = {"accuracy": 0, "completeness": 0, "relevance": 0, "overall": 0}
                print(f"❌ 查询失败: {result['error']}")
            
            # 记录结果
            test_result = {
                "phase": phase_name,
                "question_no": i,
                "question": question,
                "timestamp": datetime.now().isoformat(),
                **result,
                **evaluation
            }
            
            phase_results.append(test_result)
            self.results.append(test_result)
            
            # 避免请求过于频繁
            await asyncio.sleep(1)
        
        return phase_results
    
    async def run_full_test_suite(self):
        """执行完整的30问题测试套件"""
        print("🚀 启动RAG系统综合验证测试")
        print(f"📋 测试文档ID: {self.document_id}")
        print(f"⏰ 测试开始时间: {datetime.now()}")
        
        total_start_time = time.time()
        
        # 执行三个阶段的测试
        for phase_name, questions in self.test_questions.items():
            await self.run_test_phase(phase_name, questions)
            
        total_time = time.time() - total_start_time
        
        # 生成测试报告
        self.generate_test_report(total_time)
    
    def generate_test_report(self, total_time: float):
        """生成详细测试报告"""
        print(f"\n{'='*80}")
        print("📊 RAG系统验证测试报告")
        print(f"{'='*80}")
        
        # 基础统计
        total_questions = len(self.results)
        successful_queries = sum(1 for r in self.results if r["success"])
        failed_queries = total_questions - successful_queries
        
        # 性能统计
        response_times = [r["response_time"] for r in self.results if r["success"]]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # 质量统计
        accuracy_scores = [r["accuracy"] for r in self.results]
        completeness_scores = [r["completeness"] for r in self.results]
        relevance_scores = [r["relevance"] for r in self.results]
        overall_scores = [r["overall"] for r in self.results]
        
        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        avg_completeness = sum(completeness_scores) / len(completeness_scores)
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        avg_overall = sum(overall_scores) / len(overall_scores)
        
        print(f"📈 总体性能指标:")
        print(f"   • 总测试问题数: {total_questions}")
        print(f"   • 成功查询数: {successful_queries}")
        print(f"   • 失败查询数: {failed_queries}")
        print(f"   • 成功率: {(successful_queries/total_questions)*100:.1f}%")
        print(f"   • 平均响应时间: {avg_response_time:.2f}s")
        print(f"   • 总测试时间: {total_time:.1f}s")
        
        print(f"\n🎯 质量评估指标:")
        print(f"   • 平均准确性: {avg_accuracy:.1f}% (目标: >95%)")
        print(f"   • 平均完整性: {avg_completeness:.1f}% (目标: >90%)")
        print(f"   • 平均相关性: {avg_relevance:.1f}% (目标: >95%)")
        print(f"   • 综合得分: {avg_overall:.1f}% (目标: >90%)")
        
        # 阶段性分析
        print(f"\n📋 分阶段性能分析:")
        for phase_name in self.test_questions.keys():
            phase_results = [r for r in self.results if r["phase"] == phase_name]
            phase_success_rate = sum(1 for r in phase_results if r["success"]) / len(phase_results) * 100
            phase_avg_score = sum(r["overall"] for r in phase_results) / len(phase_results)
            print(f"   • {phase_name}: 成功率 {phase_success_rate:.1f}%, 平均得分 {phase_avg_score:.1f}%")
        
        # 性能目标达成分析
        print(f"\n🏆 目标达成度分析:")
        accuracy_target = avg_accuracy >= 95
        completeness_target = avg_completeness >= 90
        relevance_target = avg_relevance >= 95
        overall_target = avg_overall >= 90
        
        print(f"   • 准确性目标 (>95%): {'✅ 达成' if accuracy_target else '❌ 未达成'} ({avg_accuracy:.1f}%)")
        print(f"   • 完整性目标 (>90%): {'✅ 达成' if completeness_target else '❌ 未达成'} ({avg_completeness:.1f}%)")
        print(f"   • 相关性目标 (>95%): {'✅ 达成' if relevance_target else '❌ 未达成'} ({avg_relevance:.1f}%)")
        print(f"   • 综合目标 (>90%): {'✅ 达成' if overall_target else '❌ 未达成'} ({avg_overall:.1f}%)")
        
        targets_met = sum([accuracy_target, completeness_target, relevance_target, overall_target])
        print(f"   • 总体目标达成度: {targets_met}/4 ({targets_met/4*100:.1f}%)")
        
        # 保存详细结果到文件
        self.save_results_to_files()
        
        print(f"\n📄 详细结果已保存到:")
        print(f"   • JSON格式: rag_test_results.json")
        print(f"   • CSV格式: rag_test_results.csv")
        print(f"   • 详细报告: rag_test_detailed_report.txt")
    
    def save_results_to_files(self):
        """保存测试结果到多种格式文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON格式
        with open(f"rag_test_results_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # 保存CSV格式
        df = pd.DataFrame(self.results)
        df.to_csv(f"rag_test_results_{timestamp}.csv", index=False, encoding="utf-8")
        
        # 保存详细文本报告
        with open(f"rag_test_detailed_report_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write("RAG系统30问题验证测试详细报告\n")
            f.write("=" * 60 + "\n\n")
            
            for result in self.results:
                f.write(f"问题: {result['question']}\n")
                f.write(f"阶段: {result['phase']}\n")
                f.write(f"成功: {'是' if result['success'] else '否'}\n")
                f.write(f"响应时间: {result['response_time']:.2f}s\n")
                f.write(f"准确性: {result['accuracy']:.1f}%\n")
                f.write(f"完整性: {result['completeness']:.1f}%\n")
                f.write(f"相关性: {result['relevance']:.1f}%\n")
                f.write(f"综合得分: {result['overall']:.1f}%\n")
                if result['success']:
                    f.write(f"答案: {result['answer'][:200]}...\n")
                    f.write(f"来源数: {len(result['sources'])}\n")
                else:
                    f.write(f"错误: {result['error']}\n")
                f.write("-" * 50 + "\n\n")

async def main():
    """主程序入口"""
    validator = RAGSystemValidator()
    await validator.run_full_test_suite()

if __name__ == "__main__":
    asyncio.run(main())