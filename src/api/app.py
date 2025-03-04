"""
FastAPI application for the Q&A system.
"""

from typing import List, Dict, Optional
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.api.state import app_state
from src.model.llm_factory import LLMFactory

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # 设置为DEBUG级别以获取更多信息

# Initialize FastAPI app
app = FastAPI(
    title="LLM Q&A System",
    description="A Retrieval-Augmented Generation based question-answering system",
    version="0.1.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 支持的文件类型
SUPPORTED_MIME_TYPES = {
    # 文本文件
    'text/plain': '.txt',
    'text/markdown': '.md',
    'text/csv': '.csv',
    # Excel 文件
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
    'application/vnd.ms-excel': '.xls',
    # PDF 文件
    'application/pdf': '.pdf'
}

class Question(BaseModel):
    """Question request model."""
    text: str
    conversation_id: Optional[str] = None

class Answer(BaseModel):
    """Answer response model."""
    text: str
    sources: List[str]
    conversation_id: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """在应用启动时初始化应用状态"""
    try:
        if not app_state.initialize():
            logger.info("Initializing application state during startup...")
            llm = LLMFactory.create_llm()
            app_state.initialize(llm)
            logger.info("Application state initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application state: {str(e)}")
        raise

@app.post("/ask", response_model=Answer)
async def ask_question(
    question: Question,
    chain_type: str = Query("qa", description="Chain type to use (qa or conversational)")
) -> Answer:
    """Handle question requests.
    
    Args:
        question: Question request object
        chain_type: Type of chain to use (qa or conversational)
        
    Returns:
        Answer response object
    """
    try:
        if app_state.retriever is None:
            raise HTTPException(
                status_code=503,
                detail="System is initializing. Please try again later."
            )
        
        # Generate answer using specified chain_type
        result = app_state.retriever.generate_answer(question.text, chain_type=chain_type)
        
        # 处理 sources 字段，确保它是字符串列表
        sources = []
        if "sources" in result and result["sources"]:
            for source in result["sources"]:
                if hasattr(source, 'page_content'):
                    # 如果是 Document 对象，提取页面内容
                    sources.append(source.page_content[:200] + "...")  # 只取前200个字符
                elif isinstance(source, str):
                    # 如果已经是字符串，直接添加
                    sources.append(source)
                else:
                    # 其他类型，转换为字符串
                    sources.append(str(source))
        
        # 记录答案和源
        answer_text = result.get("answer", "根据找到的文档生成答案")
        logger.info(f"生成的答案: {answer_text[:100]}...")
        logger.info(f"找到 {len(sources)} 个源文档")
        
        # 确保所有文本都是有效的UTF-8编码
        def ensure_utf8(text):
            if isinstance(text, str):
                try:
                    # 尝试编码然后解码，以确保有效的UTF-8字符串
                    return text.encode('utf-8').decode('utf-8')
                except UnicodeError:
                    # 如果有错误，替换无效字符
                    return text.encode('utf-8', errors='replace').decode('utf-8')
            return str(text)
        
        answer_text = ensure_utf8(answer_text)
        sources = [ensure_utf8(s) for s in sources]
        
        return Answer(
            text=answer_text,
            sources=sources,
            conversation_id=question.conversation_id
        )
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/add")
async def add_documents(
    files: List[UploadFile] = File(...),
    doc_type: str = Query(
        "auto",
        description="Document type (auto, short_text, normal_text, long_text, code, legal, medical, chinese, english)"
    )
) -> Dict[str, int]:
    """Add documents to the knowledge base."""
    try:
        if app_state.retriever is None:
            logger.error("System initialization error: Retriever is not initialized")
            raise HTTPException(
                status_code=503,
                detail="System is initializing. Please try again later."
            )
        
        total_docs = 0
        for file in files:
            try:
                logger.info(f"开始处理文件: {file.filename}")
                logger.debug(f"文件信息 - 大小: {file.size} bytes, 类型: {file.content_type}")
                
                # 检查文件类型是否支持
                content_type = file.content_type or "text/plain"
                if content_type not in SUPPORTED_MIME_TYPES:
                    logger.warning(f"不支持的文件类型: {content_type}, 文件名: {file.filename}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file type: {content_type}"
                    )
                
                # 读取文件内容
                content = await file.read()
                logger.debug(f"成功读取文件内容，大小: {len(content)} bytes")
                
                try:
                    # 添加文档
                    logger.info(f"开始向知识库添加文档: {file.filename}")
                    app_state.retriever.add_documents(
                        documents=[content],
                        mime_type=content_type,
                        doc_type=doc_type
                    )
                    total_docs += 1
                    logger.info(f"成功处理文件: {file.filename}")
                except Exception as process_error:
                    logger.error(f"文档处理错误 - 文件: {file.filename}, 错误: {str(process_error)}", exc_info=True)
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error processing document: {str(process_error)}"
                    )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"文件处理错误 - 文件: {file.filename}, 错误: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing file {file.filename}: {str(e)}"
                )
        
        logger.info(f"成功添加文档总数: {total_docs}")
        return {"documents_added": total_docs}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"添加文档过程中发生错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 
    