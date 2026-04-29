from typing import List, Dict
from src.utils import COLOR_PALETTE, GPU_PRICES
import matplotlib.pyplot as plt

def generate_cost_report(solutions: List[Dict]) -> str:
    if not solutions:
        return "⚠️ 未找到满足条件的配置方案"
    
    report = "## 🏆 TOP3 低成本方案\n\n"
    for i, sol in enumerate(solutions, 1):
        report += (
            f'### 🔢 方案 {i}: {sol["gpu_model"]}\n'
            f'<span style="color:{COLOR_PALETTE["趋势线"]}">▎硬件配置</span>\n'
            f'- 🖥️ 单节点配置: {sol["card_count_per_node"]}卡\n'
            f'- ⚡ 单节点并发: {sol["max_conc_per_node"]}用户\n'
            f'- 🗂️ 需要节点数: {sol["nodes_required"]}个\n'
            f'- 🧮 总卡数: {sol["total_cards"]}卡\n\n'
            
            f'<span style="color:{COLOR_PALETTE["标注"]}">▎成本详情</span>\n'
            f'- 💰 单卡成本: ¥{GPU_PRICES[sol["gpu_model"]]["price"]:,.0f}\n'
            f'- 📊 总硬件成本: <span style="color:{COLOR_PALETTE["错误"]};font-weight:bold">¥{sol["total_price"]:,.2f}</span>\n'
            #f"- ⚡ 预估功耗: {sol['total_cards'] * GPU_PRICES[sol['gpu_model']]['power']}W\n"
            #f"- 🔌 电费估算: ¥{sol['total_cards'] * GPU_PRICES[sol['gpu_model']]['power'] * 0.8 * 24 * 365 / 1000:,.0f}/年\n\n"
            
            #f"<span style='color:{COLOR_PALETTE["合格"]}'>▎性价比指标</span>\n"
            #f"- 🚀 并发/卡: {sol['max_conc_per_node']/sol['card_count_per_node']:.1f} 用户/卡\n"
            #f"- 💸 成本/并发: ¥{sol['total_price']/sol['max_conc_per_node']/sol['nodes_required']:,.1f}/用户\n\n"
        )
    return report

def create_cost_chart(solutions: List[Dict]):
    """创建成本可视化图表"""
    plt.close('all')  # 修复1: 关闭之前的图表防止内存泄漏
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 设置中文字体（修复2: 显式指定字体）
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei','SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    models = [f"{s['gpu_model']}\n{s['total_cards']}卡" for s in solutions]
    prices = [s["total_price"] for s in solutions]
    
    bars = ax.bar(models, prices, color=['#4B9FFF', '#FF6F61', '#6CD9D6'])
    
    ax.set_title('TOP3 低成本方案对比')
    ax.set_ylabel('总成本 (元)')
    ax.set_xlabel('显卡配置方案')
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数据标签（修复3: 调整标签位置防止溢出）
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}',
                ha='center', va='bottom',
                fontsize=10, fontfamily='WenQuanYi Micro Hei')  # 修复4: 指定字体大小
    
    plt.tight_layout()  # 修复5: 自动调整布局
    return fig

