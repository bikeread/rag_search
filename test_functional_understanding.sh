#\!/bin/bash

# 功能理解测试脚本
API_URL="http://localhost:3001/api/query"
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbWU1OHhpbnUwMDAwa2NhMWZjdGxpaXNqIiwiZW1haWwiOiJ0ZXN0QGV4YW1wbGUuY29tIiwibmFtZSI6IlRlc3QgVXNlciIsInJvbGUiOiJVU0VSIiwiaWF0IjoxNzU0OTc3NzE1LCJleHAiOjE3NTQ5Nzg2MTV9.Jykv6LTbPX5iehfcroRYeCbO2YXe2-d-9VzIvcQ9QRI"

echo "=== RAG系统功能理解测试 ===" 
echo "开始时间: $(date)"
echo ""

# 问题1: dify_wechat_plugin的主要功能
echo "测试1: dify_wechat_plugin的主要功能"
START=$(date +%s%3N)
curl -s -X POST "$API_URL" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"dify_wechat_plugin的主要功能是什么？它如何实现微信机器人集成？"}' | jq '.'
END=$(date +%s%3N)
echo "响应时间: $((END-START))ms"
echo "---"

# 问题2: rag_search的检索增强生成
echo "测试2: rag_search的检索增强生成功能"
START=$(date +%s%3N)
curl -s -X POST "$API_URL" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"rag_search系统是如何实现检索增强生成的？包括哪些关键组件？"}' | jq '.'
END=$(date +%s%3N)
echo "响应时间: $((END-START))ms"
echo "---"

# 问题3: thesis_work_flow的论文撰写流程
echo "测试3: thesis_work_flow的工作流程"
START=$(date +%s%3N)
curl -s -X POST "$API_URL" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"thesis_work_flow如何帮助用户完成论文撰写？描述其主要工作流程"}' | jq '.'
END=$(date +%s%3N)
echo "响应时间: $((END-START))ms"
echo "---"

# 问题4: AI集成能力对比
echo "测试4: 三个项目的AI集成能力"
START=$(date +%s%3N)
curl -s -X POST "$API_URL" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"这三个项目各自是如何集成AI能力的？使用了哪些AI服务或模型？"}' | jq '.'
END=$(date +%s%3N)
echo "响应时间: $((END-START))ms"
echo "---"

# 问题5: 用户交互方式
echo "测试5: 用户交互方式对比"
START=$(date +%s%3N)
curl -s -X POST "$API_URL" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"比较这三个项目的用户交互方式，它们分别通过什么界面与用户交互？"}' | jq '.'
END=$(date +%s%3N)
echo "响应时间: $((END-START))ms"
echo "---"

# 问题6: 数据处理能力
echo "测试6: 数据处理和存储能力"
START=$(date +%s%3N)
curl -s -X POST "$API_URL" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"这些项目是如何处理和存储数据的？使用了哪些数据库或存储方案？"}' | jq '.'
END=$(date +%s%3N)
echo "响应时间: $((END-START))ms"

echo ""
echo "完成时间: $(date)"
