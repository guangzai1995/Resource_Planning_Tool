# src/utils.py

import os
import math
# 常量定义
DROPDOWN_CLASSES = ["dropdown"]
DEFAULT_MAX_DELAY = 5.0
GPU_PRICE_TABLE_HEADER = """
### 🏷️ GPU规格详情
|  型号  |  单卡租赁价格  |  产品特点       |
|:-------:|:------------:|:------------------|
"""


# 添加颜色调色板常量
COLOR_PALETTE = {
    "合格": "#4CAF50",  # 柔和的绿色
    "警告": "#FF9800",  # 温暖的橙色
    "错误": "#F44336",  # 明亮的红色
    "趋势线": "#2196F3",  # 清新的蓝色
    "插值点": "#9C27B0",  # 高贵的紫色
    "最优": "#FF4081",    # 亮眼的品红
    "背景": "#F5F5F5",    # 浅灰背景
    "标注": "#3F51B5"     # 深蓝标注
}

GPU_PRICES = {
    "4090": {"price": 85, "power": 420, "power_cost": 0.8,"desc": "消费级旗舰显卡，24GB GDDR6X显存，适合中小模型推理"}, 
    "910B1": {"price":76 , "power": 190, "power_cost": 0.8,"desc": "国产云端推理卡，64GB HBM显存，支持int4量化"},  
    "910B3": {"price": 76, "power": 210, "power_cost": 0.8,"desc":"国产高性能推理卡，适配多种推理场景，具备高效算力"}, 
    "A100": {"price": 75, "power": 270, "power_cost": 0.8, "desc": "专业计算卡/40GB显存/FP16张量核"},  
    "A40": {"price":85 , "power": 170, "power_cost": 0.8,"desc":"面向专业图形及计算场景，在多任务处理等方面表现出色"},  
    "A800": {"price":130, "power": 250, "power_cost": 0.8,"desc":"为特定计算需求优化，在大规模数据运算中发挥作用"}, 
    "H200": {"price": 320, "power": 230, "power_cost": 0.8,"desc":"新一代高性能计算卡，助力复杂计算任务高效完成"} ,  
    "H20": {"price": 170, "power": 230, "power_cost": 0.8,"desc":"新一代高性能计算卡，助力复杂计算任务高效完成"}   
}

CUSTOM_CSS = """
/* 全局字体设置 */
* {
    font-family: "Times New Roman", "SimSun", serif !important;
}

/* Gradio组件字体设置 */
.gradio-container {
    font-family: "Times New Roman", "SimSun", serif !important;
}

/* 下拉框样式 */
.dropdown select,
.gr-dropdown select,
select {
    font-family: "Times New Roman", "SimSun", serif !important;
    font-size: 14px !important;
}

.dropdown option,
.gr-dropdown option,
option {
    font-family: "Times New Roman", "SimSun", serif !important;
    font-size: 14px !important;
}

/* 文本框样式 */
.gr-textbox textarea,
.gr-textbox input,
textarea,
input[type="text"] {
    font-family: "Times New Roman", "SimSun", serif !important;
    font-size: 14px !important;
}

/* 按钮样式 */
.gr-button,
button {
    font-family: "Times New Roman", "SimSun", serif !important;
    font-size: 14px !important;
}

/* 标签样式 */
.gr-form label,
label {
    font-family: "Times New Roman", "SimSun", serif !important;
    font-size: 14px !important;
}

/* Markdown样式 */
.gr-markdown,
.markdown {
    font-family: "Times New Roman", "SimSun", serif !important;
}

/* 滑块样式 */
.gr-slider input,
.gr-slider label {
    font-family: "Times New Roman", "SimSun", serif !important;
    font-size: 14px !important;
}

/* 标签页样式 */
.gr-tabs .gr-tab-nav button {
    font-family: "Times New Roman", "SimSun", serif !important;
    font-size: 14px !important;
}

/* 图表标题和标签 */
.gr-plot {
    font-family: "Times New Roman", "SimSun", serif !important;
}

/* 针对选中的值显示 */
.gr-dropdown .wrap,
.gr-textbox .wrap {
    font-family: "Times New Roman", "SimSun", serif !important;
}

/* Token计算器样式 */
.token-calculator {
    border: 2px solid #e1e5e9;
    border-radius: 8px;
    padding: 16px;
    background-color: #f8f9fa;
    font-family: "Times New Roman", "SimSun", serif !important;
}

.token-result {
    background-color: #ffffff;
    border: 1px solid #28a745;
    border-radius: 4px;
    padding: 12px;
    font-family: "Times New Roman", monospace, serif !important;
}

/* 确保所有文本都使用指定字体 */
div, span, p, h1, h2, h3, h4, h5, h6 {
    font-family: "Times New Roman", "SimSun", serif !important;
}

/* 特定于性能报告的字体 */
.gr-textbox[data-testid="textbox"] {
    font-family: "Times New Roman", "SimSun", serif !important;
}

/* 状态文本框 */
.gr-textbox:not(.token-result) {
    font-family: "Times New Roman", "SimSun", serif !important;
}
"""

def create_gpu_price_table():
    """生成GPU价格表"""
    table = GPU_PRICE_TABLE_HEADER
    for model, info in GPU_PRICES.items():
        table += f"| {model.center(6)} | ¥{info['price']:,} | {info.get('desc', 'AI加速专用卡')} |\n"
    return table



def get_available_tokenizers():
    """获取所有可用的分词器模型路径"""
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    if not os.path.exists(model_dir):
        return []
    
    # 获取model目录下的所有子目录
    tokenizers = []
    for entry in os.scandir(model_dir):
        if entry.is_dir():
            tokenizers.append(entry.name)
    
    return tokenizers

# 添加资源计算函数
def calculate_resource_requirements(model_size, gpu_type, gpu_memory):
    """计算模型部署所需的资源需求"""
    gpu_memory = int(gpu_memory)
    
    # 根据模型参数量估算基础显存需求 (GB)
    model_memory_map = {
        "0.5B": 2,    # 0.5B模型约需2GB显存
        "7B": 18,     # 7B模型约需20GB显存
        "32B": 80,    # 32B模型约需80GB显存
        "72B": 180,   # 72B模型约需180GB显存
        "235B-MOE": 588,  # 235B MOE模型约需588GB显存
        "671B-MOE": 1678   # 671B MOE模型约需1678GB显存
    }
    
    # 获取基础显存需求
    base_memory = model_memory_map.get(model_size, 18)  # 默认7B模型
    
    # 计算最低和推荐配置的显存需求
    min_memory = base_memory
    rec_memory = base_memory * 1.5  # 推荐配置增加50%余量
    
    # 计算GPU卡数
    gpu_count_min = max(1, math.ceil(min_memory / gpu_memory))
    gpu_count_rec = max(1, math.ceil(rec_memory / gpu_memory))
    
    # 计算CPU核数 (每GPU卡对应4核最低，8核推荐)
    cpu_min = gpu_count_min * 4
    cpu_rec = gpu_count_rec * 8
    
    # 计算内存需求 (每GPU卡对应16GB最低，32GB推荐)
    memory_min = gpu_count_min * 16
    memory_rec = gpu_count_rec * 32
    
    # 硬盘需求 (固定值)
    disk_min = 100
    disk_rec = 200
    
    return [
        ["GPU卡数", str(gpu_count_min), str(gpu_count_rec)],
        ["CPU核数", str(cpu_min), str(cpu_rec)],
        ["内存 (GB)", str(memory_min), str(memory_rec)],
        ["硬盘 (GB)", str(disk_min), str(disk_rec)]
    ]