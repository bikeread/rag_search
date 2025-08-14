# RAG项目结构说明

## 项目整理日期
- **整理时间**: 2025-08-12
- **整理工具**: Claude 4 + project-organizer Subagent
- **版本**: v1.0

## 标准化目录结构

### 测试相关 `/tests/`
```
tests/
├── unit/                    # 单元测试
│   ├── api/                # API相关测试
│   ├── integration/        # 集成测试
│   ├── pages/              # 页面组件测试
│   └── python/             # Python测试文件
├── integration/            # 集成测试
├── e2e/                    # 端到端测试
│   ├── documents/          # 文档管理E2E测试
│   └── fixtures/           # E2E测试夹具
├── fixtures/               # 测试数据和夹具
└── reports/               # 测试报告
```

### 文档相关 `/docs/`
```
docs/
├── api/                    # API文档
├── guides/                 # 指导文档
│   └── REFACTOR_GUIDE.md  # 重构指南
├── reports/                # 系统报告
│   ├── RAG_SYSTEM_VALIDATION_REPORT.md
│   ├── RAG_TEST_REPORT.md
│   ├── TEST_STATUS_REPORT.md
│   └── TEST_SUMMARY.md
└── archive/               # 历史文档
```

### 数据相关 `/data/`
```
data/
├── vectors/               # 向量数据
│   ├── rag_products.faiss
│   └── rag_products.pkl
├── memory/               # 内存快照
│   └── tasks_memory_*.json
├── logs/                 # 日志文件
│   └── vector_store.log
└── temp/                 # 临时文件
```

## 测试文件分类

### 单元测试 (Unit Tests)
- **后端API测试**: `tests/unit/api/documents/*.test.ts`
  - 文档上传、删除、列表、状态检查
- **前端组件测试**: `tests/unit/pages/documents/*.test.tsx`
  - 页面组件单元测试
- **Python服务测试**: `tests/unit/python/test_*.py`
  - RAG系统核心功能测试
  - 向量化和检索测试

### 集成测试 (Integration Tests)
- **文档流程测试**: `tests/unit/integration/documents-flow.test.ts`
- **完整系统测试**: `tests/unit/python/test_complete_rag_system.py`

### 端到端测试 (E2E Tests)
- **文档管理**: `tests/e2e/documents/document-management.e2e.test.ts`
- **测试夹具**: `tests/e2e/fixtures/` (测试用文件)

## 测试数据管理

### 测试夹具 (Test Fixtures)
```
tests/fixtures/
├── test_document.txt          # 基础测试文档
├── test_ai_document.txt       # AI相关测试文档
├── test_hybrid_data.txt       # 混合数据测试
├── test_markdown.md           # Markdown测试文件
├── test_integration_doc.txt   # 集成测试文档
└── rag.txt                   # RAG系统测试数据
```

### 专业测试数据集 `test_documents/`
- **Excel数据**: `test_documents/excel/financial_data.csv`
- **技术文档**: `test_documents/tech_manual.txt`
- **用户指南**: `test_documents/user_guide.txt`
- **API文档**: `test_documents/code/api_documentation.md`

## 报告文件管理

### 测试报告 `/tests/reports/`
- **详细报告**: `rag_test_detailed_report_20250812_194056.txt`
- **结果数据**: `rag_test_results_20250812_194056.csv`
- **JSON格式**: `rag_test_results_20250812_194056.json`
- **摘要报告**: `rag_test_summary.json`

### 系统报告 `/docs/reports/`
- **系统验证**: `RAG_SYSTEM_VALIDATION_REPORT.md`
- **测试报告**: `RAG_TEST_REPORT.md`
- **状态报告**: `TEST_STATUS_REPORT.md`
- **测试摘要**: `TEST_SUMMARY.md`

## 文件命名规范

### 测试文件
- **单元测试**: `{component}.test.{ext}`
- **集成测试**: `{component}.integration.test.{ext}`
- **E2E测试**: `{component}.e2e.test.{ext}`
- **Python测试**: `test_{module}.py`

### 报告文件
- **系统报告**: `{SYSTEM}_{TYPE}_REPORT.md`
- **测试结果**: `{system}_test_{type}_{date}.{ext}`
- **摘要文件**: `{system}_summary.json`

### 数据文件
- **向量数据**: `{dataset}.{faiss|pkl}`
- **内存快照**: `tasks_memory_{timestamp}.json`
- **日志文件**: `{service}_store.log`

## 清理完成的文件

### 已整理的文件类型
- ✅ 测试文件 (*.test.*, test_*.py)
- ✅ 文档报告 (*_REPORT.md, *_GUIDE.md)
- ✅ 测试数据 (test_*.txt, test_*.md)
- ✅ 向量数据 (*.faiss, *.pkl)
- ✅ 日志文件 (*.log)
- ✅ 测试结果 (rag_test_*.*)

### 保持原位置的文件
- 📁 源代码文件 (保持在各自的src目录)
- 📁 配置文件 (保持在各服务根目录)
- 📁 Docker相关 (保持在各服务目录)
- 📁 包管理 (package.json, requirements.txt等)

## 下一步建议

1. **更新CI/CD配置**: 修改测试脚本路径
2. **更新IDE配置**: 调整测试运行器配置
3. **文档链接更新**: 更新README中的文档链接
4. **清理原始位置**: 确认整理正确后可删除原始分散文件

---
*此文档由project-organizer Subagent自动生成*