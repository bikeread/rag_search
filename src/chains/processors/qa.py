from typing import Any, Dict, List, Optional, Tuple
from src.chains.processors.base import QueryProcessor
from src.utils.logging_utils import log_args_and_result  # 导入日志装饰器
from src.data_processing.vectorization.factory import VectorizationFactory
import logging
import re
import numpy as np

class QueryIntentAnalyzer:
    """基于embedding模型的查询意图分析器"""
    
    def __init__(self, model_method="bge-m3"):
        """初始化意图分析器
        
        Args:
            model_method: 使用的向量化方法，默认为bge-m3
        """
        self.model_method = model_method
        self.vectorizer = None
        self.initialized = False
        self.logger = logging.getLogger(__name__)
        
        # 预定义的意图及其示例问题
        self.intent_examples = {
            "比较分析": ["A和B有什么区别", "比较一下X和Y的优缺点", "哪个更好"],
            "因果解释": ["为什么会出现这种情况", "产生这个问题的原因是什么", "如何解决这个问题"],
            "列举信息": ["有哪些类型", "列举常见的方法", "包含哪些步骤"],
            "概念解释": ["什么是机器学习", "人工智能的定义是什么", "这个术语是什么意思"],
            "数据统计": ["共有多少人", "平均值是多少", "增长率是多少"],
            "操作指导": ["如何操作这个功能", "使用步骤是什么", "怎么实现这个效果"]
        }
        
        # 向量化示例问题
        self.intent_vectors = {}
    
    def _init_model(self):
        """初始化向量化模型"""
        if self.initialized:
            return
            
        try:
            self.logger.info(f"初始化查询意图分析模型: {self.model_method}")
            self.vectorizer = VectorizationFactory.create_vectorizer(method=self.model_method)
            
            # 预计算各个意图的示例问题向量
            for intent, examples in self.intent_examples.items():
                example_vectors = self.vectorizer.batch_vectorize(examples)
                # 使用平均向量代表该意图
                self.intent_vectors[intent] = np.mean(np.array(example_vectors), axis=0)
                
            self.initialized = True
            self.logger.info("查询意图分析模型初始化成功")
        except Exception as e:
            self.logger.error(f"初始化查询意图分析模型失败: {str(e)}")
            raise RuntimeError(f"查询意图分析模型初始化失败: {str(e)}")
    
    def analyze_intent(self, query):
        """分析查询意图
        
        Args:
            query: 用户查询
            
        Returns:
            查询意图类型
            
        Raises:
            RuntimeError: 如果模型初始化失败或分析出错
        """
        # 确保模型已初始化
        if not self.initialized:
            try:
                self._init_model()
            except Exception as e:
                self.logger.error(f"模型初始化失败: {str(e)}")
                raise RuntimeError(f"模型初始化失败: {str(e)}")

        try:
            # 将用户查询向量化
            query_vector = self.vectorizer.vectorize(query)
            
            # 计算与各个意图的相似度
            similarities = {}
            for intent, intent_vector in self.intent_vectors.items():
                similarity = self.vectorizer.similarity(query_vector, intent_vector)
                similarities[intent] = similarity
            
            # 获取最相似的意图
            best_intent = max(similarities.items(), key=lambda x: x[1])
            self.logger.info(f"查询意图分析结果: {best_intent[0]}，相似度: {best_intent[1]:.4f}")
            
            # 如果最高相似度太低，返回一般信息查询
            if best_intent[1] < 0.5:
                return "一般信息查询"
                
            return best_intent[0]
        except Exception as e:
            self.logger.error(f"分析查询意图时出错: {str(e)}")
            raise RuntimeError(f"分析查询意图时出错: {str(e)}")


class QAQueryProcessor(QueryProcessor):
    """问答查询处理器"""
    
    def __init__(self):
        """初始化问答处理器"""
        self.intent_analyzer = QueryIntentAnalyzer()
        super().__init__()
    
    @log_args_and_result
    def process(
        self,
        query: str,
        documents: List[Any],
        chat_history: Optional[List[Tuple[str, str]]] = None,
        chain_type: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """处理问答查询"""
        import logging
        import re
        import json
        from typing import Dict, Tuple
        from src.model.llm_factory import LLMFactory
        logger = logging.getLogger(__name__)
        
        if not documents:
            logger.warning("没有找到相关文档")
            return {
                "answer": "抱歉，没有找到相关信息。",
                "sources": []
            }
        
        logger.info(f"处理查询: '{query}'，找到 {len(documents)} 个文档")
        
        try:
            # 1. 通用文档处理
            formatted_docs = []
            for i, doc in enumerate(documents):
                content = doc.page_content
                logger.info(f"文档 {i+1} 内容预览: {content[:100]}...")
                
                # 文档格式检测与预处理
                doc_format, processed_content = self._detect_and_process_format(content)
                logger.info(f"文档 {i+1} 检测到格式: {doc_format}")
                
                # 构建文档表示，包含原始内容和处理后内容
                doc_representation = {
                    "id": i+1,
                    "format": doc_format,
                    "content": processed_content,
                    "metadata": getattr(doc, 'metadata', {})
                }
                
                # 构建人类可读的文档表示
                formatted_content = self._format_document_for_llm(doc_representation)
                formatted_docs.append(formatted_content)
            
            # 2. 构建增强提示模板 
            system_prompt = """你是一个专业的信息分析助手。请根据提供的文档内容回答用户的问题。

文档处理指南:
- 文档可能包含多种格式：纯文本、表格(Markdown/CSV等)、JSON、XML、HTML等
- 对于表格数据，理解其结构、表头和内容的关系
- 对于结构化数据(如JSON/XML)，理解其层次和字段含义
- 对于列表和枚举，注意其层次关系

回答要求:
1. 仅使用提供文档中的信息，不要编造不存在的内容
2. 如果文档中没有相关信息，清晰说明"文档中未提供相关信息"
3. 回答应准确、全面且以用户容易理解的方式组织
4. 如涉及多个文档的信息，需要综合分析并避免矛盾
5. 针对不同类型的用户查询(比较、统计、具体信息等)采用适当的回答结构
6. 使用专业、客观的语气"""
            
            # 添加任务指导
            query_intent = self.intent_analyzer.analyze_intent(query)
            task_guidance = self._generate_task_guidance(query_intent)
            
            # 最终用户提示
            user_prompt = f"""问题: {query}

任务类型: {query_intent}
{task_guidance}

提供的参考文档:
{'\n\n'.join(formatted_docs)}

请基于以上文档回答问题。"""
            
            # 3. 调用语言模型
            try:
                # 使用正确的方法获取LLM实例
                llm = LLMFactory.get_llm_instance()
                logger.info(f"开始调用语言模型生成回答，查询意图: {query_intent}")
                
                # 根据是否有聊天历史决定使用哪种方式调用LLM
                if chat_history and chain_type == "conversational":
                    logger.info(f"使用对话方式，聊天历史长度: {len(chat_history)}")
                    # 格式化聊天历史
                    formatted_history = []
                    for human_msg, ai_msg in chat_history:
                        formatted_history.append({"role": "user", "content": human_msg})
                        formatted_history.append({"role": "assistant", "content": ai_msg})
                    
                    # 添加当前消息
                    messages = [
                        {"role": "system", "content": system_prompt},
                        *formatted_history,
                        {"role": "user", "content": user_prompt}
                    ]
                    # 使用chat方法生成回答
                    response = llm.chat(messages)
                else:
                    logger.info("使用普通问答方式")
                    prompt = f"{system_prompt}\n\n{user_prompt}"
                    response = llm.generate(prompt)
                
                answer = response.strip() if isinstance(response, str) else str(response).strip()
                logger.info(f"语言模型生成回答: {answer[:100]}..." if len(answer) > 100 else f"语言模型生成回答: {answer}")
            except Exception as e:
                logger.error(f"调用语言模型时出错: {str(e)}", exc_info=True)
                answer = f"抱歉，在处理您的问题时遇到了技术问题。错误信息: {str(e)}"
            
            # 4. 返回结果
            return {
                "answer": answer,
                "sources": documents
            }
        except Exception as e:
            logger.error(f"处理查询时出现未预期的错误: {str(e)}", exc_info=True)
            return {
                "answer": "抱歉，在处理您的问题时遇到了系统错误。",
                "sources": documents if documents else []
            }

    def can_handle(self, query: str) -> bool:
        """判断是否可以处理该查询"""
        # 默认可以处理所有查询
        return True 
        
    def _detect_and_process_format(self, content: str) -> Tuple[str, str]:
        """检测文档格式并进行预处理"""
        # 检测是否为表格 (Markdown或类似格式)
        if re.search(r'\|[\s-]*\|', content) and ('---' in content or '===' in content):
            return "table", content
            
        # 检测是否为JSON
        if content.strip().startswith('{') and content.strip().endswith('}'):
            try:
                import json
                # 尝试解析为JSON并美化输出
                parsed = json.loads(content)
                return "json", json.dumps(parsed, indent=2, ensure_ascii=False)
            except:
                pass
                
        # 检测是否为XML/HTML
        if re.search(r'<\w+>.*</\w+>', content, re.DOTALL) or content.strip().startswith('<'):
            return "markup", content
            
        # 检测是否为列表
        if re.search(r'^\s*[-*•]\s+', content, re.MULTILINE) or re.search(r'^\s*\d+\.\s+', content, re.MULTILINE):
            return "list", content
            
        # 默认为普通文本
        return "text", content
    
    def _format_document_for_llm(self, doc: Dict[str, Any]) -> str:
        """格式化文档为LLM可读格式"""
        # 格式化文档ID和元数据
        doc_id = doc.get('id', 'unknown')
        metadata = doc.get('metadata', {})
        content = doc.get('content', '')
        doc_format = doc.get('format', 'text')
        
        # 构建元数据字符串
        metadata_str = ""
        if metadata:
            metadata_items = []
            for k, v in metadata.items():
                if k in ['source', 'title', 'page', 'filename']:
                    metadata_items.append(f"{k}: {v}")
            if metadata_items:
                metadata_str = f"[{', '.join(metadata_items)}]"
        
        # 根据不同格式定制输出
        format_prefix = ""
        if doc_format == "table":
            format_prefix = "表格数据: "
        elif doc_format == "json":
            format_prefix = "JSON数据: "
        elif doc_format == "markup":
            format_prefix = "结构化标记: "
        elif doc_format == "list":
            format_prefix = "列表数据: "
        
        # 组合最终文档表示
        return f"文档 {doc_id} {metadata_str}\n{format_prefix}\n{content}"
    
    def _analyze_query_intent(self, query: str) -> str:
        """分析查询意图，用于优化回答策略"""
        return self.intent_analyzer.analyze_intent(query)
    
    def _generate_task_guidance(self, query_intent: str) -> str:
        """根据查询意图生成任务指导"""
        guidance_map = {
            "比较分析": "请系统性地比较相关实体之间的异同点，包括核心特征、优缺点等方面。使用对比结构组织回答。",
            "因果解释": "请详细解释原因和机制，说明因果关系，并尽可能提供具体的例证。",
            "列举信息": "请以清晰的列表形式提供所有相关项目，并简要说明每项的关键特征。",
            "概念解释": "请提供准确、全面但简洁的定义或解释，包含必要的背景信息和相关联系。",
            "数据统计": "请提供准确的数字和统计信息，如有必要可以简单解释这些数据的含义和背景。",
            "操作指导": "请以步骤形式提供清晰、具体的操作指南，确保用户能够依照指导完成操作。",
            "一般信息查询": "请提供全面、准确的信息，以清晰的结构组织回答，确保覆盖问题的各个方面。"
        }
        
        return guidance_map.get(query_intent, guidance_map["一般信息查询"])