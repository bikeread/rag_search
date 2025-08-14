#!/bin/bash

# Python服务本地启动脚本

echo "🚀 启动Python服务本地开发环境..."

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查必需的Docker服务
echo -e "${YELLOW}📋 检查依赖服务...${NC}"

check_service() {
    local service=$1
    local port=$2
    if nc -z localhost $port 2>/dev/null; then
        echo -e "${GREEN}✅ $service (端口 $port) 正在运行${NC}"
        return 0
    else
        echo -e "${RED}❌ $service (端口 $port) 未运行${NC}"
        return 1
    fi
}

# 检查各项服务
check_service "PostgreSQL" 5432
check_service "Redis" 6379
check_service "RabbitMQ" 5672
check_service "Milvus" 19530
check_service "Ollama" 11434

# 如果有服务未运行，提示启动
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}请确保所有依赖服务都在运行。你可以使用以下命令启动:${NC}"
    echo "docker-compose up -d postgres redis rabbitmq milvus ollama"
    echo ""
fi

# 设置Python路径
export PYTHONPATH="/home/bikeread/dev/rag_search/python-services/shared:/home/bikeread/dev/rag_search/python-services:$PYTHONPATH"

# 加载本地环境变量
if [ -f .env.local ]; then
    export $(cat .env.local | grep -v '^#' | xargs)
    echo -e "${GREEN}✅ 已加载本地环境变量${NC}"
else
    echo -e "${RED}❌ 未找到 .env.local 文件${NC}"
    exit 1
fi

# 检查Python虚拟环境
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 创建Python虚拟环境...${NC}"
    python3 -m venv venv
fi

# 激活虚拟环境
source venv/bin/activate
echo -e "${GREEN}✅ 已激活虚拟环境${NC}"

# 安装依赖（如果需要）
if [ ! -f "venv/installed.flag" ]; then
    echo -e "${YELLOW}📦 安装Python依赖...${NC}"
    pip install -r shared/requirements.txt
    pip install -r document-processor/requirements.txt
    pip install -r vector-service/requirements.txt
    pip install -r rag-service/requirements.txt
    touch venv/installed.flag
    echo -e "${GREEN}✅ 依赖安装完成${NC}"
fi

# 启动服务的函数
start_service() {
    local service_name=$1
    local service_dir=$2
    local port=$3
    
    echo -e "${YELLOW}🚀 启动 $service_name (端口 $port)...${NC}"
    
    # 使用新的终端窗口启动每个服务
    if command -v gnome-terminal &> /dev/null; then
        gnome-terminal --title="$service_name" -- bash -c "cd $service_dir && source ../venv/bin/activate && python main.py; exec bash"
    elif command -v konsole &> /dev/null; then
        konsole --new-tab -e bash -c "cd $service_dir && source ../venv/bin/activate && python main.py; exec bash"
    elif command -v xterm &> /dev/null; then
        xterm -title "$service_name" -e bash -c "cd $service_dir && source ../venv/bin/activate && python main.py; exec bash" &
    else
        # 如果没有GUI终端，在后台启动
        cd $service_dir && python main.py > ../$service_name.log 2>&1 &
        echo -e "${GREEN}✅ $service_name 已在后台启动 (PID: $!)${NC}"
        echo "   日志文件: $service_name.log"
    fi
}

# 选择启动模式
echo ""
echo "请选择启动模式："
echo "1) 启动所有服务"
echo "2) 只启动 Document Processor (8001)"
echo "3) 只启动 Vector Service (8002)"
echo "4) 只启动 RAG Service (8003)"
echo "5) 自定义选择"
read -p "请输入选项 (1-5): " choice

case $choice in
    1)
        start_service "Document Processor" "document-processor" 8001
        sleep 2
        start_service "Vector Service" "vector-service" 8002
        sleep 2
        start_service "RAG Service" "rag-service" 8003
        ;;
    2)
        start_service "Document Processor" "document-processor" 8001
        ;;
    3)
        start_service "Vector Service" "vector-service" 8002
        ;;
    4)
        start_service "RAG Service" "rag-service" 8003
        ;;
    5)
        read -p "启动 Document Processor? (y/n): " dp
        [ "$dp" = "y" ] && start_service "Document Processor" "document-processor" 8001
        
        read -p "启动 Vector Service? (y/n): " vs
        [ "$vs" = "y" ] && start_service "Vector Service" "vector-service" 8002
        
        read -p "启动 RAG Service? (y/n): " rs
        [ "$rs" = "y" ] && start_service "RAG Service" "rag-service" 8003
        ;;
    *)
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✨ 服务启动完成！${NC}"
echo ""
echo "📝 常用命令："
echo "  查看日志: tail -f <service-name>.log"
echo "  健康检查: curl http://localhost:8001/health"
echo "  停止服务: pkill -f 'python main.py'"
echo ""
echo "🔍 调试提示："
echo "  1. 可以在各个服务的 main.py 中添加断点"
echo "  2. 使用 VS Code 的 Python 调试器进行调试"
echo "  3. 查看 LOG_LEVEL=DEBUG 的详细日志输出"