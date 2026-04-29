### Benchmark Tools 使用说明文档

##### 基础环境要求

- Python 3.10+

##### 下载项目代码

- 点击界面的下载按钮，下载项目代码。
- 解压项目代码到本地目录。

##### 安装依赖

```shell
cd benchmark_tools
pip install -r requirements.txt
```
##### 启动测试脚本

```shell
python benchmark_parallel.py \
 --backend openai \
 --host 0.0.0.0 \
 --port 9999 \
 --tokenizer /model/Qwen2___5-0___5B-Instruct \
 --epochs 2 --parallel-num 8 \
 --prompt-tokens 2048 \
 --output-tokens 1000 \
 --served-model-name qwen \
 --api-key '123456' \
 --benchmark-csv benchmark_parallel.csv \
 --enable-prefix-caching \
 --prefix-caching-num 1024

```

##### 参数说明：

| 参数 | 类型 | 默认值 | 说明 | 
|------|------|--------|------| 
| --backend | str | 必填 | 后端类型 |
| --host | str | 127.0.0.1 | 服务端主机地址 | 
| --port | int | 9288 | 服务端端口 | 
| --tokenizer | str | 必填 | tokenizer路径 | 
| --epochs | int | 5 | 测试轮数 | 
| --parallel-num | list | [1,4,8] | 并行请求数 | 
| --prompt-tokens | list | [512,1024] | 提示词token长度 | 
| --output-tokens | list | [256,256] | 输出token长度 | 
| --served-model-name | str | None | 服务模型名称 | 
| --api-key | str | 123456 | API密钥 | 
| --benchmark-csv | str | benchmark_parallel.csv | 结果CSV文件 | 
| --enable-prefix-caching | bool | False | 是否启用前缀缓存 | 
| --prefix-caching-num | int | 0 | 前缀缓存token数 |

##### 新增多模态与并发测试参数

- 新增后端类型（chat 兼容，多模态推荐）
	- `--backend openai-chat`：OpenAI 兼容 Chat Completions 接口（/v1/chat/completions）
	- `--backend vllm-chat`：vLLM 的 OpenAI 兼容 Chat Completions 接口
- 多模态输入控制（图片 + 文本）
	- `--multimodal`：开启图文多模态请求（仅对 chat 后端生效）
	- `--random-images`：布尔开关。开启后“每个并发请求都会即时生成一张随机图片（data URL）”，无需真实图片文件。
	- `--image-size`：随机图片尺寸，默认 224x224（仅在开启 `--random-images` 时使用）

