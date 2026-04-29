
from openai import OpenAI

client = OpenAI(
    base_url="http://0.0.0.0:9999/v1",
    api_key="123456"
)

stream = client.chat.completions.create(
    model="qwen",
    messages=[
        {"role": "system", "content": "you are a helpful assistant!"},
        {"role": "user", "content": "你好"}
    ],
    stream=True
)

for chunk in stream:
    #print(chunk)
    if chunk.choices[0].delta.content is not None:
    #    print(chunk)
        print(chunk.choices[0].delta.content, end="", flush=True)
print()