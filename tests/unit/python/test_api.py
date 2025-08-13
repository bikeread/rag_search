import requests
import json

def test_ask_api():
    """测试/ask API端点"""
    url = "http://localhost:8000/ask"
    payload = {
        "text": "最贵的笔记本电脑多少钱？"
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"状态码: {response.status_code}")
        
        # 确保使用正确的编码
        response.encoding = 'utf-8'
        
        # 解析JSON响应
        if response.status_code == 200:
            result = response.json()
            print("\n回答:")
            print(result["text"])
            
            print("\n来源:")
            for i, source in enumerate(result["sources"]):
                print(f"来源 {i+1}:")
                print(source[:200] + "..." if len(source) > 200 else source)
                print()
        else:
            print(f"错误: {response.text}")
    
    except Exception as e:
        print(f"请求出错: {str(e)}")

if __name__ == "__main__":
    test_ask_api() 