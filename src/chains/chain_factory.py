"""
Chain factory module for creating different types of processing chains.
"""

import logging
from typing import Optional, Any

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain.llms.base import BaseLLM
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever

logger = logging.getLogger(__name__)

class ChainFactory:
    """Factory for creating different types of processing chains."""
    
    @staticmethod
    def create_qa_chain(
        llm: BaseLLM,
        retriever: BaseRetriever,
        chain_type: str = "stuff",
        return_source_documents: bool = True,
        verbose: bool = True,
    ) -> RetrievalQA:
        """
        创建问答链。

        Args:
            llm (BaseLLM): 语言模型
            retriever (BaseRetriever): 检索器
            chain_type (str, optional): 链类型. 默认为 "stuff".
            return_source_documents (bool, optional): 是否返回源文档. 默认为 True.
            verbose (bool, optional): 是否显示详细日志. 默认为 True.

        Returns:
            RetrievalQA: 问答链
        """
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=return_source_documents,
            verbose=verbose,
        )
        return qa_chain
    
    @staticmethod
    def create_conversational_chain(
        llm: BaseLLM,
        retriever: BaseRetriever,
        memory: Optional[ConversationBufferMemory] = None,
        prompt_template: Optional[str] = None,
        **kwargs: Any
    ) -> Chain:
        """创建会话检索链"""
        # 创建默认的对话记忆
        if memory is None:
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
        
        # 默认的对话提示模板
        default_template = """使用以下对话历史和上下文信息来回答用户的问题。如果你不知道答案，就说你不知道，不要试图编造答案。

对话历史：
{chat_history}

上下文信息：
{context}

问题：{question}

答案："""
        
        # 设置提示模板
        qa_prompt = PromptTemplate(
            template=prompt_template or default_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        # 基本链参数
        chain_kwargs = {
            "memory": memory,
            "return_source_documents": True,
            "combine_docs_chain_kwargs": {"prompt": qa_prompt},
            "verbose": True  # 启用详细日志
        }
            
        # 提取LLM特定参数直接应用到LLM，而不是传递给ConversationalRetrievalChain
        llm_kwargs = {
            k: v for k, v in kwargs.items() 
            if k in ["temperature", "max_tokens"]
        }
        
        # 如果有LLM参数，创建一个配置了这些参数的新LLM实例
        if llm_kwargs:
            try:
                # 尝试用新参数克隆LLM
                llm = llm.bind(**llm_kwargs)
            except (AttributeError, TypeError):
                # 如果不支持bind方法，记录警告
                logger.warning(
                    f"Could not apply parameters {list(llm_kwargs.keys())} to LLM. "
                    "These will be ignored."
                )
        
        # 创建对话检索链，只传递支持的参数
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            **chain_kwargs
        )
    
    @staticmethod
    def create_custom_chain(
        chain_type: str,
        llm: BaseLLM,
        retriever: BaseRetriever,
        **kwargs: Any
    ) -> Chain:
        """Create a custom chain based on type.
        
        Args:
            chain_type: Type of chain to create
            llm: Language model
            retriever: Document retriever
            **kwargs: Additional arguments for chain creation
        """
        chain_creators = {
            "qa": ChainFactory.create_qa_chain,
            "conversational": ChainFactory.create_conversational_chain,
        }
        
        creator = chain_creators.get(chain_type)
        if not creator:
            raise ValueError(f"Unsupported chain type: {chain_type}")
        
        return creator(llm=llm, retriever=retriever, **kwargs) 