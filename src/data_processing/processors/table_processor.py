"""
表格处理模块，专门处理表格数据。
"""

import os
import logging
import pandas as pd
import io
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field

from langchain.schema import Document
from .base import BaseDocumentProcessor, ProcessorConfig
from ..vectorization.factory import VectorizationFactory

logger = logging.getLogger(__name__)

@dataclass
class TableProcessorConfig(ProcessorConfig):
    """表格处理器配置"""
    max_rows_per_chunk: int = 5
    preserve_empty_cells: bool = False
    enable_vectorization: bool = False
    vectorization_method: str = field(default_factory=lambda: os.getenv('VECTORIZATION_METHOD', 'tfidf'))

class TableProcessor(BaseDocumentProcessor):
    """表格处理器实现，专门处理表格数据。"""
    
    # Excel MIME类型列表
    EXCEL_MIME_TYPES = [
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel.sheet.macroEnabled.12",
        "application/vnd.ms-excel.sheet.binary.macroEnabled.12"
    ]
    
    # CSV MIME类型列表
    CSV_MIME_TYPES = [
        "text/csv",
        "application/csv",
        "text/comma-separated-values"
    ]
    
    def __init__(self, config: Optional[TableProcessorConfig] = None):
        super().__init__(config or TableProcessorConfig())
        self.config = config or TableProcessorConfig()
        
        # 初始化向量化器
        self.vectorizer = None
        if self.config.enable_vectorization:
            try:
                self.vectorizer = VectorizationFactory.create_vectorizer(
                    method=self.config.vectorization_method
                )
                logger.info(f"表格处理器初始化向量化器: {self.config.vectorization_method}")
            except Exception as e:
                logger.error(f"初始化向量化器失败: {str(e)}")
    
    def detect_table_format(self, file_content: bytes, filename: str, mime_type: str) -> Tuple[str, Optional[pd.DataFrame]]:
        """探测表格文件类型并尝试读取为DataFrame。
        
        Args:
            file_content: 原始文件内容
            filename: 文件名
            mime_type: MIME类型
            
        Returns:
            元组: (文件类型, 可选的DataFrame)
            文件类型可以是 'csv', 'excel', 'binary_excel', 'markdown', 'text', 'unknown'
        """
        # 检查MIME类型和文件扩展名
        is_excel = mime_type in self.EXCEL_MIME_TYPES or filename.endswith(('.xls', '.xlsx', '.xlsm', '.xlsb'))
        is_csv = mime_type in self.CSV_MIME_TYPES or filename.endswith('.csv')
        
        # 如果是CSV
        if is_csv:
            try:
                df = pd.read_csv(io.BytesIO(file_content))
                return 'csv', df
            except Exception as e:
                logger.debug(f"以CSV格式读取失败: {str(e)}")
        
        # 如果是Excel
        if is_excel:
            try:
                df = pd.read_excel(io.BytesIO(file_content))
                return 'excel', df
            except Exception as e:
                logger.debug(f"以Excel格式读取失败: {str(e)}")
        
        # 尝试检测文件是否为二进制格式
        try:
            text_content = file_content.decode('utf-8', errors='strict')
            # 检查是否包含Markdown表格标记
            if "|" in text_content and "-|-" in text_content.replace(" ", ""):
                return 'markdown', None
            
            # 尝试作为CSV解析
            try:
                df = pd.read_csv(io.StringIO(text_content))
                return 'csv', df
            except Exception as e:
                logger.debug(f"文本内容作为CSV解析失败: {str(e)}")
                
            # 如果文本中包含分隔符，可能是某种表格
            if "," in text_content or "\t" in text_content:
                return 'text', None
                
            return 'text', None
            
        except UnicodeDecodeError:
            # 这是二进制内容，尝试作为Excel处理
            try:
                df = pd.read_excel(io.BytesIO(file_content))
                return 'binary_excel', df
            except Exception as excel_err:
                logger.debug(f"二进制内容作为Excel处理失败: {str(excel_err)}")
                return 'unknown', None
    
    def process_file(self, file_content: bytes, filename: str, mime_type: str) -> List[Document]:
        """处理表格文件。
        
        Args:
            file_content: 原始文件内容
            filename: 文件名
            mime_type: MIME类型
            
        Returns:
            处理后的文档列表
        """
        try:
            logger.debug(f"开始处理表格文件: {filename}, MIME类型: {mime_type}")
            
            # 使用探测方法确定文件类型
            file_type, df = self.detect_table_format(file_content, filename, mime_type)
            logger.debug(f"检测到文件类型: {file_type}")
            
            if df is not None:
                # 已经成功读取为DataFrame
                logger.debug(f"已成功读取为DataFrame，列数: {len(df.columns)}, 行数: {len(df)}")
                return self._process_dataframe(df, {"source": filename, "file_type": file_type})
            
            # 如果没有成功读取为DataFrame，根据检测到的类型处理
            if file_type == 'markdown':
                logger.debug(f"检测到Markdown表格格式，使用Markdown解析器处理")
                text_content = file_content.decode('utf-8', errors='ignore')
                return self._process_markdown_table(text_content, {"source": filename})
            elif file_type in ['text', 'unknown']:
                # 尝试作为文本处理
                logger.debug(f"尝试将内容作为普通文本处理")
                text_content = file_content.decode('utf-8', errors='ignore')
                return self.process_text(text_content, {"source": filename})
            else:
                # 其他情况，也尝试作为文本处理
                logger.warning(f"无法确定文件类型，尝试作为文本处理")
                text_content = file_content.decode('utf-8', errors='ignore')
                return self.process_text(text_content, {"source": filename})
            
        except Exception as e:
            logger.error(f"处理表格文件时出错 {filename}: {str(e)}", exc_info=True)
            return []
    
    def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """处理表格文本内容。
        
        Args:
            text: 要处理的文本内容
            metadata: 可选的元数据
            
        Returns:
            处理后的文档列表
        """
        try:
            logger.debug(f"开始处理表格文本，文本长度: {len(text)}")
            logger.debug(f"文本前100个字符: {text[:100]}")
            
            # 尝试解析Markdown表格
            if "|" in text:
                logger.debug("检测到可能的Markdown表格格式，尝试解析Markdown表格")
                return self._process_markdown_table(text, metadata or {})
            
            # 尝试解析CSV格式
            try:
                logger.debug("尝试将文本解析为CSV格式")
                # 先显示一下即将解析的文本前几行，帮助排查问题
                if len(text.split('\n')) > 1:
                    first_few_lines = '\n'.join(text.split('\n')[:3])
                    logger.debug(f"CSV解析前文本前几行: {first_few_lines}")
                
                df = pd.read_csv(io.StringIO(text))
                logger.debug(f"成功解析为CSV，列数: {len(df.columns)}, 行数: {len(df)}")
                logger.debug(f"列名: {df.columns.tolist()}")
                return self._process_dataframe(df, metadata or {})
            except Exception as csv_err:
                logger.debug(f"解析为CSV失败: {str(csv_err)}")
            
            # 如果无法解析为表格，返回原始文本
            logger.warning("无法将文本解析为表格，返回原始文本")
            doc = Document(
                page_content=text,
                metadata=metadata or {}
            )
            
            # 如果启用了向量化，添加向量
            if self.vectorizer and self.config.enable_vectorization:
                doc.metadata["vector"] = self.vectorizer.vectorize(text)
                
            return [doc]
            
        except Exception as e:
            logger.error(f"处理表格文本时出错: {str(e)}", exc_info=True)
            return []
    
    def _process_dataframe(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> List[Document]:
        """处理DataFrame并分块。
        
        Args:
            df: 要处理的DataFrame
            metadata: 元数据
            
        Returns:
            处理后的文档列表
        """
        documents = []
        
        logger.debug(f"开始处理DataFrame，原始大小: {df.shape}")
        logger.debug(f"列名: {df.columns.tolist()}")
        
        # 处理空值
        if not self.config.preserve_empty_cells:
            df = df.fillna("")
        
        # 获取列名
        headers = df.columns.tolist()
        
        # 按指定行数分块
        for i in range(0, len(df), self.config.max_rows_per_chunk):
            chunk_df = df.iloc[i:i+self.config.max_rows_per_chunk]
            logger.debug(f"处理DataFrame分块 {i//self.config.max_rows_per_chunk + 1}/{(len(df) + self.config.max_rows_per_chunk - 1)//self.config.max_rows_per_chunk}, 大小: {chunk_df.shape}")
            
            # 转换为Markdown表格
            markdown_table = self._dataframe_to_markdown(chunk_df)
            
            # 创建文档
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": i // self.config.max_rows_per_chunk,
                "total_chunks": (len(df) + self.config.max_rows_per_chunk - 1) // self.config.max_rows_per_chunk,
                "row_start": i,
                "row_end": min(i + self.config.max_rows_per_chunk, len(df)),
                "columns": headers
            })
            
            doc = Document(
                page_content=markdown_table,
                metadata=chunk_metadata
            )
            
            # 如果启用了向量化，添加向量
            if self.vectorizer and self.config.enable_vectorization:
                doc.metadata["vector"] = self.vectorizer.vectorize(markdown_table)
            
            documents.append(doc)
        
        logger.debug(f"DataFrame处理完成，生成了 {len(documents)} 个文档")
        return documents
    
    def _process_markdown_table(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """处理Markdown表格文本。
        
        Args:
            text: Markdown表格文本
            metadata: 元数据
            
        Returns:
            处理后的文档列表
        """
        lines = text.strip().split("\n")
        logger.debug(f"开始处理Markdown表格，总行数: {len(lines)}")
        
        if len(lines) < 2:
            logger.warning(f"Markdown表格行数不足，无法解析: {len(lines)}")
            return [Document(page_content=text, metadata=metadata)]
        
        # 提取表头
        header_line = lines[0].strip()
        headers = [h.strip() for h in header_line.strip("|").split("|")]
        logger.debug(f"解析到表头: {headers}, 列数: {len(headers)}")
        
        # 跳过分隔行
        data_start_idx = 1
        if len(lines) > 1 and set(lines[1].strip()) <= {"-", "+", "|"}:
            data_start_idx = 2
            logger.debug("检测到Markdown表格分隔行，从第3行开始解析数据")
        
        # 提取数据行
        data_rows = []
        for i in range(data_start_idx, len(lines)):
            if lines[i].strip():
                cells = [cell.strip() for cell in lines[i].strip("|").split("|")]
                logger.debug(f"第{i+1}行解析结果: {cells}, 列数: {len(cells)}")
                
                # 检查是否列数匹配
                if len(cells) != len(headers):
                    logger.warning(f"第{i+1}行列数与表头不匹配: 表头{len(headers)}列, 数据行{len(cells)}列")
                    # 如果列数不匹配，可以尝试修正（填充或截断）
                    if len(cells) < len(headers):
                        logger.debug(f"列数不足，填充空值")
                        cells.extend([''] * (len(headers) - len(cells)))
                    else:
                        logger.debug(f"列数过多，截断到表头列数")
                        cells = cells[:len(headers)]
                
                data_rows.append(cells)
        
        logger.debug(f"解析完成，数据行数: {len(data_rows)}")
        
        try:
            # 创建DataFrame
            logger.debug(f"尝试创建DataFrame，数据行数: {len(data_rows)}, 列数: {len(headers)}")
            # 输出前几行数据，帮助诊断
            if data_rows:
                logger.debug(f"第一行数据: {data_rows[0]}")
            
            df = pd.DataFrame(data_rows, columns=headers)
            logger.debug(f"DataFrame创建成功，形状: {df.shape}")
            
            # 处理DataFrame
            return self._process_dataframe(df, metadata)
        except Exception as e:
            logger.error(f"创建DataFrame失败: {str(e)}", exc_info=True)
            # 如果发生1列错误，尝试修复
            if "1 columns passed, passed data had" in str(e):
                logger.debug("尝试修复列数不匹配问题...")
                try:
                    # 确保所有行的长度一致，以最大长度为准
                    max_cols = max(len(row) for row in data_rows)
                    logger.debug(f"检测到的最大列数: {max_cols}")
                    
                    # 重新生成列名，确保列名长度匹配数据行
                    if len(headers) != max_cols:
                        logger.debug(f"调整列名数量从 {len(headers)} 到 {max_cols}")
                        # 如果表头列数少于数据列数，则扩展表头
                        if len(headers) < max_cols:
                            headers.extend([f"Column{i+1}" for i in range(len(headers), max_cols)])
                        # 如果表头列数多于数据列数，则截断表头
                        else:
                            headers = headers[:max_cols]
                    
                    # 确保每行数据长度一致
                    uniform_data = []
                    for row in data_rows:
                        if len(row) < max_cols:
                            row.extend([''] * (max_cols - len(row)))
                        elif len(row) > max_cols:
                            row = row[:max_cols]
                        uniform_data.append(row)
                    
                    logger.debug(f"修复后的数据行数: {len(uniform_data)}")
                    df = pd.DataFrame(uniform_data, columns=headers)
                    logger.debug(f"修复后DataFrame创建成功，形状: {df.shape}")
                    return self._process_dataframe(df, metadata)
                except Exception as repair_error:
                    logger.error(f"尝试修复列数不匹配失败: {str(repair_error)}")
            
            # 如果修复失败，返回原始文本
            return [Document(page_content=text, metadata=metadata)]
    
    def _dataframe_to_markdown(self, df: pd.DataFrame) -> str:
        """将DataFrame转换为Markdown表格。
        
        Args:
            df: 要转换的DataFrame
            
        Returns:
            Markdown表格文本
        """
        # 获取列名
        headers = df.columns.tolist()
        
        # 构建表头
        header_row = "| " + " | ".join(str(h) for h in headers) + " |"
        
        # 构建分隔行
        separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        
        # 构建数据行
        data_rows = []
        for _, row in df.iterrows():
            data_row = "| " + " | ".join(str(cell) for cell in row) + " |"
            data_rows.append(data_row)
        
        # 组合所有行
        markdown_table = "\n".join([header_row, separator_row] + data_rows)
        
        return markdown_table 