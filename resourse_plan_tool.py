
import os
import gradio as gr
import matplotlib.pyplot as plt
from src.data_loader import DataLoader  
from src.performance import Performance   
from src.token_processor import InitTokenizer,  TokenProcessor 
from src.run_benchmark import start_benchmark,create_benchmark_tab
from src.visualization import generate_cost_report, create_cost_chart
from src.utils import  CUSTOM_CSS,DROPDOWN_CLASSES,DEFAULT_MAX_DELAY,create_gpu_price_table,get_available_tokenizers
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 初始化分词器
try:
    my_tokenizer = InitTokenizer(model_path="./model/Qwen2___5-0___5B-Instruct")
    tokenizer=my_tokenizer.tokenizer
    tokenizer_available =my_tokenizer.tokenizer_available
    mytoken_processor=TokenProcessor()
except Exception as e:
    print(f"初始化分词器失败: {e}") 
    tokenizer = None
    tokenizer_available = False

# 加载初始数据
performance_data=DataLoader()
current_performance_data = performance_data.performance_data
gpu_models, model_names, card_counts, input_lengths = performance_data.get_available_options()
performance=Performance(current_performance_data)

# 辅助：将结构化结果格式化为 Markdown 供 Gradio 展示
def _format_perf_result(res):
    try:
        if isinstance(res, dict):
            if "optimal" in res:
                opt = res.get("optimal", {})
                bnd = res.get("boundary", {})
                lines = [
                    "## 📊 性能分析报告",
                    "",
                    f"- 最大并发: `{opt.get('concurrency')}`",
                    f"- 系统总吞吐量: `{opt.get('total_throughput'):.2f}` tokens/s" if isinstance(opt.get('total_throughput'), (int,float)) else f"- 系统总吞吐量: `{opt.get('total_throughput')}`",
                    f"- 单用户吞吐量: `{opt.get('per_user_throughput'):.2f}` tokens/s" if isinstance(opt.get('per_user_throughput'), (int,float)) else f"- 单用户吞吐量: `{opt.get('per_user_throughput')}`",
                    f"- 是否使用插值: `{opt.get('interpolation_used')}`",
                ]
                if bnd:
                    lines += [
                        f"- 边界并发(延时约束): `{bnd.get('max_acceptable_concurrency')}`",
                        f"- 边界估算总吞吐: `{bnd.get('estimated_total_throughput')}`",
                    ]
                return "\n".join(lines)
            if "message" in res:
                return f"**{res.get('message')}**"
        return str(res)
    except Exception:
        return str(res)

def _format_token_stats(stats):
    try:
        if isinstance(stats, dict):
            return (
                f"📋 Token 统计报告\n\n"
                f"- 系统提示词: {stats.get('system_tokens', 0)} tokens ({stats.get('system_chars', 0)} 字符)\n\n"
                f"- 用户输入: {stats.get('user_tokens', 0)} tokens ({stats.get('user_chars', 0)} 字符)\n\n"
                f"🔺 总计: {stats.get('total_tokens', 0)} tokens ({stats.get('total_chars', 0)} 字符)\n"
            )
        return str(stats)
    except Exception:
        return str(stats)

# 界面组件封装函数
def create_status_components():
    """创建状态显示组件"""
    with gr.Row():
        refresh_btn = gr.Button("🔄 刷新数据", variant="secondary")
        status_text = gr.Textbox(
            label="状态", 
            value=f"已加载 {len(gpu_models)} 个GPU型号", 
            interactive=False
        )
    return refresh_btn, status_text

def create_performance_tab():
    """创建性能计算器标签页"""
    gr.Markdown("## 🎯 最大并发计算工具")
    gr.Markdown("选择显卡配置和模型参数，计算最大并发用户数和性能指标。")

    with gr.Row():
        with gr.Column():
            refresh_btn = gr.Button("🔄 刷新数据", variant="secondary")
            status_text = gr.Textbox(
                label="状态", 
                value=f"已加载 {len(gpu_models)} 个GPU型号", 
                interactive=False
            )
            gpu_model = gr.Dropdown(
                label="选择GPU型号",
                choices=gpu_models,
                value=gpu_models[0] if gpu_models else None,
                elem_classes=DROPDOWN_CLASSES
            )
            model_name = gr.Dropdown(
                label="选择模型名称",
                choices=[],
                value=None,
                elem_classes=DROPDOWN_CLASSES
            )
            card_count = gr.Dropdown(
                label="选择卡数",
                choices=[],
                value=None,
                elem_classes=DROPDOWN_CLASSES
            )
            input_length = gr.Dropdown(
                label="输入长度 (tokens)",
                choices=[],
                value=None,
                elem_classes=DROPDOWN_CLASSES
            )
            output_length = gr.Slider(
                label="输出长度 (tokens)",
                minimum=128,
                maximum=8192,
                step=128,
                value=1200
                # elem_classes=SLIDER_CLASSES
            )
            max_delay = gr.Slider(
                label="最大首token延时要求 (秒)",
                minimum=0.1,
                maximum=10,
                step=0.1,
                value=DEFAULT_MAX_DELAY
            )
            # 添加单用户吞吐量Slider
            user_throughput = gr.Slider(
                label="单用户吞吐量要求 (tokens/s)",
                minimum=1,
                maximum=30,
                step=1,
                value=10
            )
            calc_btn = gr.Button("🚀 计算最大并发用户数", variant="primary")
        
        with gr.Column():
            performance_chart = gr.Plot(label="📊 性能图表")
            output_report = gr.Markdown(
                label="📋 性能报告", 
                elem_classes=["token-result"]
            )
    
    return gpu_model, model_name, card_count, input_length, max_delay, calc_btn, performance_chart, output_report ,refresh_btn, status_text,user_throughput,output_length

def create_token_tab():
    """创建Token计算器标签页"""
    gr.Markdown("## 📏 输入长度估算工具")
    gr.Markdown("通过分词器精确计算系统提示词和用户输入的token数量，帮助您选择合适的输入长度进行性能测试。")
    
    with gr.Row():
        with gr.Column():
            example_dropdown = gr.Dropdown(
                label="📝 选择示例",
                choices=[""] + [ex["name"] for ex in mytoken_processor.get_token_examples()],
                value="",
                elem_classes=DROPDOWN_CLASSES
            )
            
            # 添加分词器选择下拉框
            tokenizer_dropdown = gr.Dropdown(
                label="🔤 选择分词器",
                choices=get_available_tokenizers(),
                #value="./model/Qwen2___5-0___5B-Instruct",
                elem_classes=DROPDOWN_CLASSES
            )

            system_prompt_input = gr.Textbox(
                label="🤖 系统提示词 (System Prompt)",
                placeholder="请输入系统提示词...",
                lines=4,
                max_lines=10
            )
            
            user_prompt_input = gr.Textbox(
                label="👤 用户输入 (User Prompt)",
                placeholder="请输入用户提示词...",
                lines=6,
                max_lines=15
            )
            
            with gr.Row():
                calculate_tokens_btn = gr.Button("🔢 计算Token数量", variant="primary")
                clear_btn = gr.Button("🗑️ 清空输入", variant="secondary")
        
        with gr.Column():
            token_result = gr.Markdown(
                label="📊 Token统计结果",
                elem_classes=["token-result"]
            )
            
            # 显示分词器信息
            # tokenizer_info = gr.Markdown(
            #     f"**🔧 分词器状态**: {'✅ 已加载 - ' + (tokenizer.name_or_path if tokenizer_available and tokenizer else '❌ 未加载，使用字符计数模式')}"
            # )
            # 注意：此处 tokenizer_dropdown 组件尚未完成渲染，其 value 可能为 None，
            # 不能直接传给 os.path.basename，否则会触发 TypeError。
            if tokenizer_available and tokenizer:
                try:
                    base_name = os.path.basename(get_available_tokenizers()[0]) if get_available_tokenizers() else "未知分词器"
                    init_status = f"✅ 已加载 - {base_name}"
                except Exception:
                    init_status = "✅ 已加载"
            else:
                init_status = "❌ 未加载，使用字符计数模式"

            tokenizer_info = gr.Markdown(
                f"**🔧 分词器状态**: {init_status}"
            )
    
    return example_dropdown, system_prompt_input, user_prompt_input, calculate_tokens_btn, clear_btn, token_result,tokenizer_dropdown,tokenizer_info

def create_cost_tab():
    """创建成本核算器标签页"""
    gr.Markdown("## 📊 部署成本计算工具")
    gr.Markdown("根据性能需求计算最优硬件配置方案")
    
    with gr.Row():
        with gr.Column():
            cost_concurrency = gr.Number(label="目标最大并发用户数", minimum=1, value=50)
            cost_model = gr.Dropdown(label="选择模型", choices=model_names, value=model_names[-1])
            cost_length = gr.Dropdown(label="上下文长度", choices=input_lengths, value=input_lengths[1])
            cost_delay = gr.Slider(label="最大首token延时 (秒)", minimum=0.1, maximum=10, value=DEFAULT_MAX_DELAY, step=0.1)
            cost_calc_btn = gr.Button("计算最优方案", variant="primary")
            
            # GPU价格表
            gr.Markdown(create_gpu_price_table())
        
        with gr.Column():
            cost_chart = gr.Plot(label="成本对比图表")
            cost_report = gr.Markdown(label="最优方案详情", elem_classes=["token-result"])
    
    return cost_concurrency, cost_model, cost_length, cost_delay, cost_calc_btn, cost_chart, cost_report



def bind_events(refresh_btn, status_text, gpu_model, model_name, card_count, 
                input_length, calc_btn, output_report, performance_chart, 
                example_dropdown, system_prompt_input, user_prompt_input, 
                calculate_tokens_btn, clear_btn, token_result, cost_concurrency, 
                cost_model, cost_length, cost_delay, cost_calc_btn, 
                cost_chart, cost_report,user_throughput,output_length,
                run_btn, backend, host, port, tokenizer, epochs, parallel_num, prompt_tokens, output_tokens, served_model_name, api_key, enable_prefix_caching, prefix_caching_num, benchmark_csv, log_output, result_file,
                tokenizer_dropdown,tokenizer_info
                ):


    """绑定所有界面事件"""
    # 性能计算器事件绑定
    # 修复: 原先只更新 model_name, 若新 GPU 下存在同名模型则不会触发 model_name.change, 导致 card_count / input_length 仍为旧 GPU 数据
    # 现在切换 GPU 时直接级联返回 (model, card, input_length) 三个下拉选项，避免例如 H200 仍显示上一张卡的单一 512 输入长度问题
    gpu_model.change(
        fn=lambda g: performance_data.initialize_cascading_updates(g),
        inputs=[gpu_model],
        outputs=[model_name, card_count, input_length]
    )
    
    # 当模型变化时同时更新卡数和输入长度，避免只有单一卡数时不触发后续 input_length 更新导致仍显示旧 GPU/模型的单一输入长度（例如 H200 仅显示 512）
    model_name.change(
        fn=lambda g, m: (
            performance_data.update_card_choices(g, m),
            # 需要先拿到新的卡数选项的默认值，再计算输入长度
            performance_data.update_input_length_choices(
                g,
                m,
                performance_data.update_card_choices(g, m).get('value') if isinstance(performance_data.update_card_choices(g, m), dict) else None
            )
        ),
        inputs=[gpu_model, model_name],
        outputs=[card_count, input_length]
    )
    
    card_count.change(
        fn=performance_data.update_input_length_choices,
        inputs=[gpu_model, model_name, card_count],
        outputs=[input_length]
    )
    
    refresh_btn.click(
        fn=performance_data.refresh_data,
        outputs=[gpu_model, model_name, card_count, input_length, status_text]
    )
    
    def _calc_and_format(g, m, c, i, d, ut, ol):
        res, fig, = performance.calculate_max_concurrency_optimized(g, m, c, i, d, ut, ol)[0:2]
        return _format_perf_result(res), fig

    calc_btn.click(
        fn=_calc_and_format,
        inputs=[gpu_model, model_name, card_count, input_length, max_delay, user_throughput, output_length],
        outputs=[output_report, performance_chart]
    )
    
    # 分词器选择事件
    tokenizer_dropdown.change(
        fn=lambda path: (
            my_tokenizer.set_model_path(path),
            f"**🔧 分词器状态**: {'✅ 已加载 - ' + os.path.basename(path) if my_tokenizer.tokenizer_available else '❌ 加载失败'}"
        ),
        inputs=[tokenizer_dropdown],
        outputs=[gr.State(), tokenizer_info]
    )
    
    # 更新Token计算事件
    def _calc_tokens_and_format(sys_p, usr_p):
        stats = my_tokenizer.calculate_input_tokens(sys_p, usr_p)
        return _format_token_stats(stats)

    calculate_tokens_btn.click(
        fn=_calc_tokens_and_format,
        inputs=[system_prompt_input, user_prompt_input],
        outputs=[token_result]
    )
    # Token计算器事件绑定
    # calculate_tokens_btn.click(
    #     fn=my_tokenizer.calculate_input_tokens,
    #     inputs=[system_prompt_input, user_prompt_input],
    #     outputs=[token_result]
    # )
    
    clear_btn.click(
        fn=lambda: ("", "", ""),
        outputs=[system_prompt_input, user_prompt_input, token_result]
    )
    
    example_dropdown.change(
        fn=mytoken_processor.load_example,
        inputs=[example_dropdown],
        outputs=[system_prompt_input, user_prompt_input]
    )
    
    # 成本核算器事件绑定
    cost_calc_btn.click(
        fn=lambda mc, m, cl, md: (
            create_cost_chart(performance.calculate_cost_optimization(mc, m, cl, md)),
            generate_cost_report(performance.calculate_cost_optimization(mc, m, cl, md))
        ),
        inputs=[cost_concurrency, cost_model, cost_length, cost_delay],
        outputs=[cost_chart, cost_report]
    )

    run_btn.click(
        fn=start_benchmark,
        inputs=[backend, host, port, tokenizer, epochs, parallel_num, prompt_tokens, output_tokens, served_model_name, api_key, enable_prefix_caching, prefix_caching_num, benchmark_csv],
        outputs=[log_output, result_file]
    )
# 创建Gradio界面
with gr.Blocks(title="GPU并发计算器", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
    gr.Markdown("# 🚀 资源规划工具")
    gr.Markdown("根据GPU卡性能计算可支持的最大并发用户数（每用户10 tokens/s）")
    
    with gr.Tabs():
        with gr.TabItem("📈 最大并发计算器"):
            gpu_model, model_name, card_count, input_length, max_delay, calc_btn, performance_chart, output_report ,refresh_btn, status_text ,user_throughput,output_length = create_performance_tab()

        with gr.TabItem("🧮 Token计算器"):
            example_dropdown, system_prompt_input, user_prompt_input, calculate_tokens_btn, clear_btn, token_result,tokenizer_dropdown,tokenizer_info = create_token_tab()

        
        with gr.TabItem("💰 成本核算器"):
            cost_concurrency, cost_model, cost_length, cost_delay, cost_calc_btn, cost_chart, cost_report = create_cost_tab()
        
        with gr.TabItem("⚡ 性能测试工具"):
            run_btn, backend, host, port, tokenizer, epochs, parallel_num, prompt_tokens, output_tokens, served_model_name, api_key, enable_prefix_caching, prefix_caching_num, benchmark_csv, log_output, result_file = create_benchmark_tab()

    # 绑定事件处理
    bind_events(
        refresh_btn, status_text, gpu_model, model_name, card_count, input_length, calc_btn, output_report, performance_chart, 
        example_dropdown, system_prompt_input, user_prompt_input, calculate_tokens_btn, clear_btn, token_result, 
        cost_concurrency, cost_model, cost_length, cost_delay, cost_calc_btn, cost_chart, cost_report,user_throughput,output_length,
        run_btn, backend, host, port, tokenizer, epochs, parallel_num, prompt_tokens, output_tokens, served_model_name, api_key, enable_prefix_caching, prefix_caching_num, benchmark_csv, log_output, result_file,
        tokenizer_dropdown,tokenizer_info

    )
    
    # 界面加载完成后初始化级联更新
    if gpu_models:
        demo.load(
            fn=lambda: performance_data.initialize_cascading_updates(gpu_models[0]),
            outputs=[model_name, card_count, input_length]
        )

# 启动应用
if __name__ == "__main__":
    demo.queue()
    demo.launch(
        share=False,
        server_port=15421,
        server_name='0.0.0.0',
        show_error=True
    )

