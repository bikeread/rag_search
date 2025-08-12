#\!/bin/bash

API_URL="http://localhost:3001/api/query"
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbWU1OHhpbnUwMDAwa2NhMWZjdGxpaXNqIiwiZW1haWwiOiJ0ZXN0QGV4YW1wbGUuY29tIiwibmFtZSI6IlRlc3QgVXNlciIsInJvbGUiOiJVU0VSIiwiaWF0IjoxNzU0OTc3NzE1LCJleHAiOjE3NTQ5Nzg2MTV9.Jykv6LTbPX5iehfcroRYeCbO2YXe2-d-9VzIvcQ9QRI"

echo "=== 功能理解测试结果 ==="
echo ""

questions=(
  "dify_wechat_plugin的主要功能是什么？它如何实现微信机器人集成？"
  "rag_search系统是如何实现检索增强生成的？包括哪些关键组件？"
  "thesis_work_flow如何帮助用户完成论文撰写？描述其主要工作流程"
  "这三个项目各自是如何集成AI能力的？使用了哪些AI服务或模型？"
  "比较这三个项目的用户交互方式，它们分别通过什么界面与用户交互？"
  "这些项目是如何处理和存储数据的？使用了哪些数据库或存储方案？"
)

for i in {0..5}; do
  echo "问题$((i+1)): ${questions[$i]}"
  echo "---"
  START=$(date +%s%3N)
  
  response=$(curl -s -X POST "$API_URL" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"${questions[$i]}\"}")
  
  END=$(date +%s%3N)
  
  echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
  echo "响应时间: $((END-START))ms"
  echo "============================================"
  echo ""
  
  sleep 1
done

echo "测试完成: $(date)"
