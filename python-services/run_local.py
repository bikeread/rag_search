#!/usr/bin/env python3
"""
Python服务本地调试启动器
使用方法: python run_local.py [service_name]
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# 服务配置
SERVICES = {
    'document-processor': {
        'dir': 'document-processor',
        'port': 8001,
        'env': {
            'HOST': '0.0.0.0',
            'PORT': '8001',
            'RABBITMQ_URL': 'amqp://guest:guest@localhost:5672',
            'VECTOR_SERVICE_URL': 'http://localhost:8002'
        }
    },
    'vector-service': {
        'dir': 'vector-service',
        'port': 8002,
        'env': {
            'HOST': '0.0.0.0',
            'PORT': '8002',
            'MILVUS_HOST': 'localhost',
            'MILVUS_PORT': '19530',
            'MILVUS_COLLECTION': 'rag_documents',
            'VECTOR_DIM': '384'
        }
    },
    'rag-service': {
        'dir': 'rag-service',
        'port': 8003,
        'env': {
            'HOST': '0.0.0.0',
            'PORT': '8003',
            'VECTOR_SERVICE_URL': 'http://localhost:8002',
            'OLLAMA_URL': 'http://localhost:11434',
            'OLLAMA_MODEL': 'llama3.2:1b'
        }
    }
}

def check_port(port):
    """检查端口是否被占用"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result == 0

def start_service(name, config):
    """启动单个服务"""
    print(f"\n🚀 启动 {name} (端口 {config['port']})...")
    
    # 检查端口
    if check_port(config['port']):
        print(f"⚠️  端口 {config['port']} 已被占用，跳过启动")
        return None
    
    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{Path.cwd()}/shared:{Path.cwd()}"
    env['LOG_LEVEL'] = 'DEBUG'
    env.update(config['env'])
    
    # 启动服务
    service_dir = Path.cwd() / config['dir']
    process = subprocess.Popen(
        [sys.executable, 'main.py'],
        cwd=service_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    print(f"✅ {name} 已启动 (PID: {process.pid})")
    return process

def main():
    """主函数"""
    print("=" * 50)
    print("Python服务本地调试启动器")
    print("=" * 50)
    
    # 解析参数
    if len(sys.argv) > 1:
        service_name = sys.argv[1]
        if service_name in SERVICES:
            # 启动指定服务
            process = start_service(service_name, SERVICES[service_name])
            if process:
                try:
                    # 实时输出日志
                    for line in process.stdout:
                        print(line, end='')
                except KeyboardInterrupt:
                    print(f"\n⏹️  停止 {service_name}...")
                    process.terminate()
        else:
            print(f"❌ 未知服务: {service_name}")
            print(f"可用服务: {', '.join(SERVICES.keys())}")
    else:
        # 启动所有服务
        processes = []
        for name, config in SERVICES.items():
            p = start_service(name, config)
            if p:
                processes.append((name, p))
            time.sleep(2)  # 等待服务启动
        
        if processes:
            print("\n" + "=" * 50)
            print("✨ 所有服务已启动！")
            print("=" * 50)
            print("\n📝 健康检查:")
            time.sleep(3)
            
            for name, _ in processes:
                port = SERVICES[name]['port']
                url = f"http://localhost:{port}/health"
                try:
                    import requests
                    response = requests.get(url, timeout=2)
                    if response.status_code == 200:
                        print(f"✅ {name}: 健康")
                    else:
                        print(f"⚠️  {name}: 状态异常")
                except:
                    print(f"❌ {name}: 无响应")
            
            print("\n按 Ctrl+C 停止所有服务...")
            
            try:
                # 等待中断
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n⏹️  停止所有服务...")
                for name, p in processes:
                    p.terminate()
                    print(f"✅ {name} 已停止")

if __name__ == '__main__':
    main()