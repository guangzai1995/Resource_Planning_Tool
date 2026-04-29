# import requests
# import json

# # API配置
# url = "http://10.86.12.11:32510/ai-paas/ai-open/sitech/aiopen/stream/Jiusi-32B-Chat-AWQ-TEST/v1/completions"
# #url="http://10.200.2.202:13446/v1/completions"
# headers = {
#     "Content-Type": "application/json",
#     "Authorization": "Bearer 1lfFS1uynW84LX8JuYwjTsQ="  # API密钥格式需要保持一致
# }

# # 请求数据
# data = {
#     "model": "Jiusi-32B-AWQ",
#     "prompt":"xxxxxxxxxxasd Ada ef",
#     "stream": True,  # 启用流式响应
#     "max_tokens":100
#     }

# response = requests.post(url, headers=headers, json=data, stream=True)
# response.raise_for_status()  # 检查请求是否成功

# # 处理流式响应
# for line in response.iter_lines():
#     print(line.decode('utf-8'))

import requests
import json

# API配置
url = "http://10.86.12.11:32510/ai-paas/ai-open/sitech/aiopen/stream/Jiusi-32B-Chat-AWQ-TEST/v1/completions"
#url="http://10.200.2.202:13446/v1/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer lfFS1uynW84LX8JuYwjTsQ="  # 注意这里你之前写的是1lf而不是llf
}

# 请求数据
data = {
    "model": "Jiusi-32B-AWQ",
    "prompt":"xxxxxxxxxxasd Ada ef",
    "stream": True,  # 启用流式响应
    "max_tokens":100
}

try:
    # 发送请求
    response = requests.post(url, headers=headers, json=data, stream=True)
    
    # 打印实际发送的请求头部（调试用）
    print("实际发送的请求头部：")
    for key, value in response.request.headers.items():
        print(f"{key}: {value}")
    
    response.raise_for_status()  # 检查请求是否成功

    # 处理流式响应
    for line in response.iter_lines():
        if line:  # 过滤空行
            print(line.decode('utf-8'))
except Exception as e:
    print(f"请求发生错误：{e}")