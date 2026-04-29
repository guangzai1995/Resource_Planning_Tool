### Benchmark Tools 使用说明文档

<!-- #### 目录结构

```bash

benchmark_tools/
├── api-test.py               # API测试工具
├── benchmark.py              # 主基准测试脚本
├── benchmark_parallel.py     # 并行基准测试脚本
├── benchmark_serving.py      # 服务端基准测试脚本
├── benchmark_utils.py        # 基准测试工具函数
├── generate_dataset.py       # 数据集生成工具
├── modal_benchmark/          # 多模态基准测试工具
├── multilora_benchmark/      # 多LoRA基准测试工具
├── prefix_run.sh             # 参数化运行脚本
├── requirements.txt          # 依赖包列表
├── run.sh                    # 示例运行脚本
└── speculative_benchmark_parallel.py  # 推测解码基准测试脚本
``` -->
<!-- #### 使用说明 -->
  
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
| --api-key | str | (可选) | API密钥（可选；不传则不加 Authorization 头；也可通过环境变量 `BENCHMARK_API_KEY`/`API_KEY`/`OPENAI_API_KEY` 提供） | 
| --benchmark-csv | str | benchmark_parallel.csv | 结果CSV文件 | 
| --enable-prefix-caching | bool | False | 是否启用前缀缓存 | 
| --prefix-caching-num | int | 0 | 前缀缓存token数 |
