
import os
import time
import queue
import threading
import subprocess
import gradio as gr
def enqueue_output(out, queue):
    """将子进程输出放入队列"""
    for line in iter(out.readline, ''):
        queue.put(line)
    out.close()


def run_benchmark(
    backend, host, port, tokenizer, epochs, parallel_num, 
    prompt_tokens, output_tokens, served_model_name, api_key, 
    enable_prefix_caching, prefix_caching_num, benchmark_csv
):
    """执行性能测试并返回日志"""
    log = []
    # 创建日志文件路径
    log_file = os.path.join("benchmark_tools", "benchmark.log")
    
    # 清空并初始化日志文件
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("开始性能测试...\n")
    log.append("开始性能测试...")

    # 构建命令参数
    cmd = [
        "bash", "prefix_run.sh",
        "--backend", f"{backend}",
        "--host", f"{host}",
        "--port", f"{int(port)}",
        "--tokenizer", f"{tokenizer}",
        "--epochs", f"{int(epochs)}",
        "--parallel-num", f"{parallel_num}",
        "--prompt-tokens", f"{prompt_tokens}",
        "--output-tokens", f"{output_tokens}",
        "--served-model-name", f"{served_model_name}",
        "--api-key", f"{api_key}",
        "--benchmark-csv", f"{benchmark_csv}",
    ]
    if enable_prefix_caching:
        cmd.extend([
            "--enable-prefix-caching", f"{'True'}",
            "--prefix-caching-num", f"{int(prefix_caching_num)}"
        ])
    try:
        # 执行命令并捕获输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd="benchmark_tools"
        )

        # 实时读取输出并写入日志文件
        with open(log_file, 'a', encoding='utf-8') as f:
            for line in iter(process.stdout.readline, ''):
                log_line = line.strip()
                log.append(log_line)
                f.write(log_line + '\n')
                f.flush()  # 确保内容立即写入文件
                time.sleep(0.01)

        process.wait()
        final_log = f"测试完成，退出码: {process.returncode}\n结果文件: {benchmark_csv}"
        log.append(final_log)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(final_log + '\n')

        # 构建完整的CSV文件路径
        csv_path = os.path.join("benchmark_tools", benchmark_csv)
        return '\n'.join(log), csv_path if os.path.exists(csv_path) else None

    except Exception as e:
        error_msg = f"测试失败: {str(e)}"
        log.append(error_msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(error_msg + '\n')
        return '\n'.join(log), None
    

# 添加日志文件路径
LOG_FILE_PATH = os.path.join("benchmark_tools", "benchmark.log")

def run_benchmark_thread(*args, result_queue):
    """在后台线程中执行性能测试"""
    try:
        log, csv_file = run_benchmark(*args)
        result_queue.put((log, csv_file))
    except Exception as e:
        result_queue.put((f"执行错误: {str(e)}", None))

def start_benchmark(*args):
    """启动性能测试并初始化日志"""
    
    # 提取验证所需的参数
    # args顺序：backend, host, port, tokenizer, epochs, parallel_num, prompt_tokens, output_tokens, served_model_name, api_key, enable_prefix_caching, prefix_caching_num, benchmark_csv
    prompt_tokens = args[6]
    output_tokens = args[7]
    enable_prefix_caching = args[10]
    prefix_caching_num = args[11]
    
    # 执行验证
    token_error = validate_token_counts(prompt_tokens, output_tokens)
    if token_error:
        yield token_error, None
        return
    
    prefix_error = validate_prefix_caching(enable_prefix_caching, prefix_caching_num, prompt_tokens)
    if prefix_error:
        yield prefix_error, None
        return
    
    # 获取用户指定的结果文件名（最后一个参数）
    original_filename = args[-1]
    # 分割文件名和扩展名，添加时间戳后缀
    #name, ext = os.path.splitext(original_filename)
    ext=".csv"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    new_filename = f"{original_filename}_{timestamp}{ext}"
    # 更新参数列表中的文件名
    new_args = list(args[:-1]) + [new_filename]
    result_file_path = os.path.join("benchmark_tools", new_filename)


    # 清空日志文件
    with open(LOG_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write("开始性能测试...\n")
    
    # 创建结果队列
    result_queue = queue.Queue()
    
    # 启动后台测试线程
    thread = threading.Thread(
        target=run_benchmark_thread,
        args=new_args,
        kwargs={"result_queue": result_queue},
        daemon=True
    )
    thread.start()
    
    # 定期读取日志文件并更新界面
    while thread.is_alive():
        if os.path.exists(LOG_FILE_PATH):
            with open(LOG_FILE_PATH, 'r', encoding='utf-8') as f:
                log_content = f.read()
            # 返回日志内容以更新界面
            yield log_content, None
        time.sleep(1)
    
    # 测试完成，获取最终结果
    log, csv_file = result_queue.get()
    
    # 优化：添加文件存在性检测和重试机制
    max_retries = 3
    retry_count = 0
    file_found = False
    
    # 循环检查文件是否存在，处理可能的文件生成延迟
    while retry_count < max_retries and not file_found:
        if os.path.exists(result_file_path):
            # 验证文件大小，确保文件已完全写入
            if os.path.getsize(result_file_path) > 0:
                csv_file = result_file_path
                file_found = True
                log += f"\n✅ 测试结果文件已生成: {result_file_path}"
                break
            else:
                log += f"\n⚠️ 检测到空文件，等待重试 ({retry_count+1}/{max_retries})..."
        else:
            log += f"\n⚠️ 未找到结果文件，等待重试 ({retry_count+1}/{max_retries})..."
        
        retry_count += 1
        time.sleep(1)  # 等待1秒后重试
        yield log, None  # 更新日志显示
    
    if not file_found:
        log += f"\n❌ 测试结果文件生成失败，请检查测试配置"
        csv_file = None

    yield log, csv_file

def validate_token_counts(prompt_tokens, output_tokens):
    """验证输入长度和输出长度的数量是否一致"""
    prompt_list = prompt_tokens.strip().split() if prompt_tokens.strip() else []
    output_list = output_tokens.strip().split() if output_tokens.strip() else []
    
    if len(prompt_list) != len(output_list):
        return f"⚠️ 错误：输入长度数量（{len(prompt_list)}）与输出长度数量（{len(output_list)}）必须一致！"
    
    try:
        for p in prompt_list:
            int(p)
        for o in output_list:
            int(o)
        return ""
    except ValueError:
        return "⚠️ 错误：输入必须为整数！"

def validate_prefix_caching(enable_prefix, prefix_num, prompt_tokens):
    """验证前缀缓存数量是否超过输入长度"""
    if not enable_prefix:
        return ""
    
    if not prompt_tokens.strip():
        return "⚠️ 错误：请先输入输入长度！"
    
    try:
        prompt_list = list(map(int, prompt_tokens.strip().split()))
        prefix_num = int(prefix_num)
        
        for p in prompt_list:
            if prefix_num > p:
                return f"⚠️ 错误：前缀缓存数量（{prefix_num}）不能超过输入长度（{p}）！"
        
        return ""
    except ValueError:
        return "⚠️ 错误：输入长度或前缀缓存数量必须为整数！"

def create_benchmark_tab():
    """创建性能测试工具标签页"""
    gr.Markdown("## ⚡ 性能测试工具")
    gr.Markdown("配置测试参数并执行性能测试，查看实时日志并下载结果文件。")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 参数配置")
            backend = gr.Dropdown(
                label="后端类型 (--backend)",
                choices=["openai"],
                value="openai",
                info="选择服务后端类型"
            )
            host = gr.Textbox(
                label="服务地址 (--host)",
                value="0.0.0.0",
                info="测试服务的IP地址"
            )
            port = gr.Number(
                label="服务端口 (--port)",
                value=9999,
                precision=0,
                info="测试服务的端口号"
            )
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_file_dir)
            model_dir = os.path.join(project_root, 'model')
            model_folders = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]
            model_paths = [os.path.join(model_dir, folder) for folder in model_folders]
            
            tokenizer_choices = [(folder, path) for folder, path in zip(model_folders, model_paths)]
            tokenizer = gr.Dropdown(
                choices=tokenizer_choices,
                label="分词器路径 (--tokenizer)",
                value=tokenizer_choices[0][1] if tokenizer_choices else "",
                info="选择测试模型分词器，主要用于计算token数量"
            )
            epochs = gr.Number(
                label="测试轮次 (--epochs)",
                value=2,
                precision=0,
                info="测试执行的轮次数量"
            )
            parallel_num = gr.Textbox(
                label="并行数 (--parallel-num)",
                value="8",
                info="并行测试的进程数量，多个值用空格分隔（例如：1 4 8）"
            )
            prompt_tokens = gr.Textbox(
                label="输入长度 (--prompt-tokens)",
                value="2048",
                info="提示词的token数量，多个值用空格分隔（例如：1024 2048 4096）"
            )
            output_tokens = gr.Textbox(
                label="输出长度 (--output-tokens)",
                value="500",
                info="生成结果的token数量，多个值用空格分隔（例如：500 1000 500）"
            )
            
            served_model_name = gr.Textbox(
                label="模型名称 (--served-model-name)",
                value="qwen",
                info="服务端模型名称"
            )
            api_key = gr.Textbox(
                label="API密钥 (--api-key)",
                value="123456",
                info="访问API的密钥"
            )
            enable_prefix_caching = gr.Checkbox(
                label="启用前缀缓存 (--enable-prefix-caching)",
                value=True,
                info="并发数据中存在相同的系统提示词时开启，模拟toB场景，（可以降低首token延时）"
            )
            prefix_caching_num = gr.Number(
                label="缓存数量 (--prefix-caching-num)",
                value=1024,
                precision=0,
                info="相同系统提示词的token数量，若未启用前缀缓存则无需配置"
            )
            # 添加复选框事件，控制缓存数量输入框的可见性和交互性
            enable_prefix_caching.change(
                fn=lambda x: gr.update(visible=x, interactive=x),
                inputs=enable_prefix_caching,
                outputs=prefix_caching_num
            )
            benchmark_csv = gr.Textbox(
                label="结果文件名 (--benchmark-csv)",
                value="benchmark_parallel",
                info="测试结果CSV文件名，以csv格式+时间戳保存"
            )
            error_message = gr.Markdown(label="验证信息", value="", visible=True)
            run_btn = gr.Button("▶️ 开始测试", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 测试日志")
            #gr.HTML("<style>.log-output { background-color: #000000; color: #ffffff; }</style>")
            log_output = gr.Textbox(
                label="执行日志",
                lines=20,
                interactive=False
                #elem_classes=["log-output"]
            )
            gr.Markdown("### 结果下载")
            result_file = gr.File(label="测试结果CSV")
            zip_file = gr.File(
                label="工具包下载",
                value="benchmark_tools.zip",
                interactive=False
            )
            # 读取工具包使用说明
            readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark_tools", "README.md")
            with open(readme_path, "r", encoding="utf-8") as f:
                readme_content = f.read()

            # 显示工具包使用说明
            #gr.Markdown("### 工具包使用说明")
            gr.Markdown(readme_content)
    
    # 添加验证事件绑定
    prompt_tokens.change(
        fn=validate_token_counts,
        inputs=[prompt_tokens, output_tokens],
        outputs=[error_message]
    )
    
    output_tokens.change(
        fn=validate_token_counts,
        inputs=[prompt_tokens, output_tokens],
        outputs=[error_message]
    )
    
    enable_prefix_caching.change(
        fn=validate_prefix_caching,
        inputs=[enable_prefix_caching, prefix_caching_num, prompt_tokens],
        outputs=[error_message]
    )
    
    prefix_caching_num.change(
        fn=validate_prefix_caching,
        inputs=[enable_prefix_caching, prefix_caching_num, prompt_tokens],
        outputs=[error_message]
    )

    return run_btn, backend, host, port, tokenizer, epochs, parallel_num, prompt_tokens, output_tokens, served_model_name, api_key, enable_prefix_caching, prefix_caching_num, benchmark_csv, log_output, result_file
