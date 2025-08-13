# RAG系统测试套件

## 目录结构

```
tests/
├── unit/                    # 单元测试
│   ├── api/                # 后端API测试
│   ├── integration/        # 集成测试
│   ├── pages/              # 前端组件测试
│   └── python/             # Python服务测试
├── integration/            # 跨服务集成测试
├── e2e/                    # 端到端测试
├── fixtures/               # 测试数据和夹具
└── reports/               # 测试报告

```

## 测试运行命令

### 前端测试
```bash
# 在frontend目录运行
cd frontend
npm run test

# 特定测试文件
npm run test -- DocumentListPage.test.tsx
```

### 后端测试
```bash
# 在backend目录运行
cd backend
npm run test

# API测试
npm run test -- --grep "documents"
```

### Python测试
```bash
# 在项目根目录运行
cd tests/unit/python
python -m pytest test_*.py

# 特定测试
python test_complete_rag_system.py
```

### 端到端测试
```bash
# 在项目根目录运行
npx playwright test tests/e2e/

# 特定E2E测试
npx playwright test tests/e2e/documents/
```

## 测试数据说明

### fixtures/ 目录
包含各种测试用的数据文件：
- `test_document.txt` - 基础文档测试
- `test_ai_document.txt` - AI功能测试
- `test_hybrid_data.txt` - 混合检索测试
- `test_markdown.md` - Markdown解析测试

### 测试配置
- 测试环境变量配置在各服务的配置文件中
- 数据库和向量存储使用测试专用实例
- 所有测试数据都是模拟数据，不包含敏感信息

## 报告查看

测试报告存储在 `tests/reports/` 目录：
- CSV格式：便于数据分析
- JSON格式：程序化处理
- TXT格式：详细日志

## 注意事项

1. 运行测试前确保所有服务都已启动
2. 测试数据会自动清理，无需手动处理
3. E2E测试需要完整的系统环境
4. 集成测试可能需要较长时间运行