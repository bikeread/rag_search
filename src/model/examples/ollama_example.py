from src.model.ollama_client import OllamaClient
import os
from dotenv import load_dotenv

def test_ollama():
    # 加载环境变量
    load_dotenv()
    
    # 方式1：使用环境变量中的配置
    ollama_client = OllamaClient()
    
    # 方式2：使用自定义参数覆盖环境变量配置
    # ollama_client = OllamaClient(
    #     model_name="codellama",
    #     temperature=0.8,
    #     max_tokens=4096
    # )
    
    # 确保模型已下载
    try:
        ollama_client.load_model()
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return
    
    # 测试简单的对话
    prompts = [
        "What is artificial intelligence?",
        "Write a Python function to calculate fibonacci numbers.",
    ]
    
    for prompt in prompts:
        try:
            print(f"\nPrompt: {prompt}")
            response = ollama_client(prompt)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Failed to generate response: {str(e)}")

if __name__ == "__main__":
    test_ollama() 