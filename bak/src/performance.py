import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
from src.utils import COLOR_PALETTE, GPU_PRICES

class Performance:
    def __init__(self, performance_data):
        self.current_performance_data= performance_data
    def create_optimized_performance_chart(self, gpu_model, model_name, card_count, input_length, 
                                        max_first_token_delay, optimal_concurrency, optimal_throughput, user_throughput=10.0):


        """创建优化后的性能图表，突出显示最优配置"""
        plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'  # 显式设置字体
        plt.rcParams['axes.unicode_minus'] = False
        plt.close('all')
        
        try:
            # 获取数据
            concurrency_data = self.current_performance_data[gpu_model][model_name][card_count][input_length]
            sorted_data = sorted(concurrency_data.items(), key=lambda x: x[0])
            
            concurrencies = [item[0] for item in sorted_data]
            throughputs = [item[1]["throughput"] for item in sorted_data]
            first_token_delays = [item[1]["first_token"] / 1000.0 for item in sorted_data]
            
            # 判断每个点的状态（使用严格小于逻辑）
            point_colors = []
            point_labels = []
            
            for i, (conc, throughput, delay) in enumerate(zip(concurrencies, throughputs, first_token_delays)):
                per_user = throughput / conc
                
                if delay >= max_first_token_delay:  # 严格小于，所以>=为不满足
                    point_colors.append('#FF6B6B')  # 柔和的红色
                    point_labels.append('延时不满足(>=要求)')
                #elif per_user < 10.0:
                elif per_user < user_throughput:

                    point_colors.append('#FFB86C')  # 柔和的橙色
                    point_labels.append('单用户吞吐量不足')
                else:
                    point_colors.append('#50FA7B')  # 柔和的绿色
                    point_labels.append('满足所有要求')
            
            # 创建图表 - 使用两个子图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # 第一个子图：并发数 vs 吞吐量
            scatter1 = ax1.scatter(concurrencies, throughputs, c=point_colors, s=60, alpha=0.7, label='原始数据点')
            ax1.plot(concurrencies, throughputs, '#6272A4', alpha=0.3, linewidth=1, label='趋势线')  # 柔和的蓝色
            
            # 添加插值点标记
            interpolation_points = []
            
            # 检查是否使用了插值计算
            interpolation_used = False
            max_acceptable_concurrency = None
            estimated_throughput = None
            
            # 重新计算插值点以便在图表中显示
            for i in range(len(first_token_delays)):
                if first_token_delays[i] >= max_first_token_delay:
                    if i > 0:
                        x1, y1 = concurrencies[i-1], first_token_delays[i-1]
                        x2, y2 = concurrencies[i], first_token_delays[i]
                        
                        epsilon = 0.001
                        target_delay = max_first_token_delay - epsilon
                        
                        if y2 != y1:
                            estimated_concurrency = x1 + (target_delay - y1) * (x2 - x1) / (y2 - y1)
                            if x1 <= estimated_concurrency <= x2:
                                t1, t2 = throughputs[i-1], throughputs[i]
                                estimated_throughput = t1 + (estimated_concurrency - x1) * (t2 - t1) / (x2 - x1)
                                max_acceptable_concurrency = estimated_concurrency
                                interpolation_used = True
                                
                                # 添加插值点到图表
                                ax1.scatter(estimated_concurrency, estimated_throughput, 
                                        c='#8BE9FD', s=150, marker='s', alpha=0.8,  # 柔和的青色
                                        edgecolors='#6272A4', linewidth=2, 
                                        label='延时约束插值点', zorder=4)
                                
                                interpolation_points.append((estimated_concurrency, estimated_throughput, target_delay))
                    break
            else:
                # 所有点都满足延时要求，进行外推
                # 所有点都满足延时要求时，不进行外推
                if len(sorted_data) >= 1:
                    # 直接使用最后一个数据点
                    max_acceptable_concurrency = concurrencies[-1]
                    estimated_throughput = throughputs[-1]
                    interpolation_used = True
                    
                    # 添加最后一个数据点作为参考点
                    ax1.scatter(max_acceptable_concurrency, estimated_throughput, 
                            c='#8BE9FD', s=150, marker='s', alpha=0.8,
                            edgecolors='#6272A4', linewidth=2, 
                            label='最大实测点', zorder=4)
                            
                            #interpolation_points.append((estimated_concurrency, estimated_throughput, target_delay))
            
            # 计算最优配置的正确纵坐标值
            if optimal_concurrency and optimal_throughput:
                # 判断最优点是否为插值点
                is_interpolated = abs(optimal_concurrency - (max_acceptable_concurrency or 0)) < 0.01
                
                # 计算第一个子图的正确纵坐标（吞吐量）
                optimal_y1 = optimal_throughput  # 直接使用计算好的插值结果
                
                # 计算第二个子图的正确纵坐标（延时）
                optimal_y2 = None
                for i in range(len(concurrencies) - 1):
                    if concurrencies[i] <= optimal_concurrency <= concurrencies[i + 1]:
                        # 在相邻数据点之间进行延时插值
                        x1, x2 = concurrencies[i], concurrencies[i + 1]
                        d1, d2 = first_token_delays[i], first_token_delays[i + 1]
                        optimal_y2 = d1 + (optimal_concurrency - x1) * (d2 - d1) / (x2 - x1)
                        break
                
                # 如果找不到相邻点，使用边界值
                if optimal_y2 is None:
                    if optimal_concurrency >= concurrencies[-1]:
                        optimal_y2 = first_token_delays[-1]
                    else:
                        optimal_y2 = first_token_delays[0]
                
                point_type = "插值最优解" if is_interpolated else "数据点最优解"
                
                # 在第一个子图中显示最优点
                ax1.scatter(optimal_concurrency, optimal_y1, 
                        c='#FF79C6', s=300, marker='*',  # 柔和的粉色
                        edgecolors='#BD93F9', linewidth=2,  # 柔和的紫色边框
                        label='最优配置', zorder=5)
                
                ax1.annotate(f'最优: 并发{int(optimal_concurrency)}\n吞吐量{optimal_y1:.1f} tokens/s\n{int(optimal_concurrency)}用户\n({point_type})', 
                            (optimal_concurrency, optimal_y1),
                            xytext=(15, 15), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', fc='#FF79C6', alpha=0.2),
                            fontsize=9, weight='bold')
                
                # 在第二个子图中显示最优点
                ax2.scatter(concurrencies, first_token_delays, c=point_colors, s=60, alpha=0.7, label='原始数据点')
                ax2.plot(concurrencies, first_token_delays, '#6272A4', alpha=0.3, linewidth=1, label='趋势线')
                
                # 添加延时要求线（严格小于）
                ax2.axhline(y=max_first_token_delay, color='#FF6B6B', linestyle='--', linewidth=2,
                        label=f'延时要求: < {max_first_token_delay}s (严格小于)', alpha=0.8)
                
                # 添加插值点的延时标记（不是最优配置点）
                for conc, throughput, delay in interpolation_points:
                    ax2.scatter(conc, delay, 
                            c='#8BE9FD', s=150, marker='s', alpha=0.8,
                            edgecolors='#6272A4', linewidth=2, 
                            label='延时约束插值点', zorder=4)
                
                # 添加最优配置点（与插值点区分）
                ax2.scatter(optimal_concurrency, optimal_y2, 
                        c='#FF79C6', s=300, marker='*', 
                        edgecolors='#BD93F9', linewidth=2,
                        label='最优配置', zorder=5)
                
                # 添加延时数值标注
                ax2.annotate(f'延时: {optimal_y2:.3f}s', 
                            (optimal_concurrency, optimal_y2),
                            xytext=(15, -15), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', fc='#FF79C6', alpha=0.2),
                            fontsize=9, weight='bold')
            else:
                # 如果没有最优配置，正常绘制第二个子图
                ax2.scatter(concurrencies, first_token_delays, c=point_colors, s=60, alpha=0.7, label='原始数据点')
                ax2.plot(concurrencies, first_token_delays, '#6272A4', alpha=0.3, linewidth=1, label='趋势线')
                
                # 添加延时要求线（严格小于）
                ax2.axhline(y=max_first_token_delay, color='#FF6B6B', linestyle='--', linewidth=2,
                        label=f'延时要求: < {max_first_token_delay}s (严格小于)', alpha=0.8)
                
                # 添加插值点的延时标记
                for conc, throughput, delay in interpolation_points:
                    ax2.scatter(conc, delay, 
                            c='#8BE9FD', s=150, marker='s', alpha=0.8,
                            edgecolors='#6272A4', linewidth=2, 
                            label='延时约束插值点', zorder=4)
            
            ax1.set_xlabel('并发数')
            ax1.set_ylabel('总吞吐量 (tokens/s)')
            ax1.set_title(f'{gpu_model} - {model_name} - {card_count}卡 - 输入长度{input_length} (优化分析)')
            ax1.grid(True, alpha=0.3)
            
            # 添加图例到第一个子图
            from matplotlib.patches import Patch
            legend_elements = []
            
            # 添加插值点图例
            if interpolation_used:
                if any('外推' in str(p) for p in interpolation_points):
                    legend_elements.append(
                        Patch(facecolor='#8BE9FD', alpha=0.7, label='延时约束外推点')
                    )
                else:
                    legend_elements.append(
                        Patch(facecolor='#8BE9FD', alpha=0.7, label='延时约束插值点')
                    )
            
            # 添加最优配置图例
            if optimal_concurrency and optimal_throughput:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='*', color='#FF79C6', 
                            markersize=15, 
                            linestyle='None', 
                            markeredgewidth=1, label='最优配置')
                )
            
            ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)
            
            ax2.set_xlabel('并发数')
            ax2.set_ylabel('首token延时 (秒)')
            ax2.set_title('首token延时随并发数变化 (严格小于要求)')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
            
            # 添加阴影区域表示可接受的延时范围
            ax2.axhspan(0, max_first_token_delay, alpha=0.1, color='#50FA7B', label='可接受延时范围')
            
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            # 错误处理
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'优化图表生成错误: {str(e)}', ha='center', va='center', fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig

    def calculate_max_concurrency_optimized(self,gpu_model, model_name, card_count, 
                                                input_length, max_first_token_delay,
                                                user_throughput=10.0,output_length=1200):

        """
        优化的最大并发用户数计算逻辑
        1. 根据延时要求通过线性插值找到最大可接受并发数（严格小于延时要求）
        2. 检查该并发数下的单用户吞吐量是否满足≥10 tokens/s
        3. 如不满足，则进一步计算满足条件的最大并发数
        """
        current_performance_data=self.current_performance_data
        try:
            # 检查数据是否存在
            if (not gpu_model or not model_name or not card_count or not input_length or
                gpu_model not in current_performance_data or 
                model_name not in current_performance_data[gpu_model] or
                card_count not in current_performance_data[gpu_model][model_name] or
                input_length not in current_performance_data[gpu_model][model_name][card_count]):
                return "错误：未找到所选配置的性能数据", None
            
            # 获取该配置下的所有并发数据，按并发数排序
            concurrency_data = current_performance_data[gpu_model][model_name][card_count][input_length]
            sorted_data = sorted(concurrency_data.items(), key=lambda x: x[0])  # 按并发数排序
            
            if len(sorted_data) == 0:
                return "错误：没有性能数据", None
            
            # 提取数据点
            concurrencies = [item[0] for item in sorted_data]
            # 根据output_length决定使用数据吞吐量还是公式计算
            if output_length == 1200:
                throughputs = [item[1]["throughput"] for item in sorted_data]
                #print(f"吞吐量测试：{throughputs}")

            else:
                throughputs = []
                for item in sorted_data:
                    conc = item[0]
                    data = item[1]
                    first_token_s = data["first_token"] / 1000.0
                    incremental_delay_s = data["incremental_delay"] / 1000.0
                    # 使用公式计算估算吞吐量
                    estimated_throughput = (output_length * conc) / (
                        first_token_s + incremental_delay_s * output_length
                    )
                    throughputs.append(estimated_throughput)
                    #print(f"吞吐量测试：{throughputs}")
            #throughputs = [item[1]["throughput"] for item in sorted_data]
            first_token_delays = [item[1]["first_token"] / 1000.0 for item in sorted_data]  # 转换为秒
            
            # 第1步：通过线性插值找到最接近延时要求的并发数（严格小于延时要求）
            max_acceptable_concurrency = None
            estimated_throughput = None
            interpolation_used = False
            
            # 找到第一个超过延时要求的点进行插值
            for i in range(len(first_token_delays)):
                if first_token_delays[i] >= max_first_token_delay:
                    if i > 0:
                        # 在第i-1和第i个点之间进行线性插值
                        x1, y1 = concurrencies[i-1], first_token_delays[i-1]
                        x2, y2 = concurrencies[i], first_token_delays[i]
                        
                        # 计算延时刚好小于要求时的并发数
                        # 使用 max_first_token_delay - epsilon 确保严格小于
                        epsilon = 0.001  # 小量，确保严格小于
                        target_delay = max_first_token_delay - epsilon
                        
                        if y2 != y1:  # 避免除零
                            # 线性插值公式: x = x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)
                            estimated_concurrency = x1 + (target_delay - y1) * (x2 - x1) / (y2 - y1)
                            
                            # 确保插值结果在合理范围内
                            if x1 <= estimated_concurrency <= x2:
                                max_acceptable_concurrency = estimated_concurrency
                                
                                # 同时插值计算对应的吞吐量
                                t1, t2 = throughputs[i-1], throughputs[i]
                                estimated_throughput = t1 + (estimated_concurrency - x1) * (t2 - t1) / (x2 - x1)
                                interpolation_used = True
                                break
                        
                        # 如果无法插值，使用前一个点
                        if not interpolation_used:
                            max_acceptable_concurrency = x1
                            estimated_throughput = throughputs[i-1]
                            break
                    else:
                        # 第一个点就超过要求，无法插值
                        return "警告: 最小并发数的延时都超过了要求！", None,0
                    break
            else:
                # 所有点都满足延时要求，找到最大的并发数并进行边界插值
                if first_token_delays:
                    last_delay = first_token_delays[-1]
                    last_conc = concurrencies[-1]
                    last_throughput = throughputs[-1]
                    
                    if last_delay < max_first_token_delay:
                        # 直接使用最后一个数据点
                        max_acceptable_concurrency = last_conc
                        estimated_throughput = last_throughput
                        interpolation_used = False  # 标记为未使用插值
                    else:
                        return "警告: 数据异常，无法计算满足延时要求的并发数！", None,0
            
            if max_acceptable_concurrency is None or estimated_throughput is None:
                return "警告: 无法计算满足延时要求的并发数！", None,0
            
            # 第2步：检查单用户吞吐量是否满足≥10 tokens/s
            per_user_throughput = estimated_throughput / max_acceptable_concurrency
            
            #if per_user_throughput >= 10.0:
            if per_user_throughput >= user_throughput:

                optimal_concurrency = int(max_acceptable_concurrency)
                
                # 统一通过插值计算吞吐量（无论是否存在该并发数的数据点）
                optimal_throughput = None
                for i in range(len(concurrencies) - 1):
                    if concurrencies[i] <= optimal_concurrency <= concurrencies[i + 1]:
                        # 在相邻数据点之间进行线性插值
                        x1, x2 = concurrencies[i], concurrencies[i + 1]
                        t1, t2 = throughputs[i], throughputs[i + 1]
                        optimal_throughput = t1 + (optimal_concurrency - x1) * (t2 - t1) / (x2 - x1)
                        break
                
                # 如果找不到相邻点（例如并发数超出数据范围），使用最后一个点的值
                if optimal_throughput is None:
                    if optimal_concurrency >= concurrencies[-1]:
                        optimal_throughput = throughputs[-1]
                    else:
                        optimal_throughput = throughputs[0]
                
                optimal_per_user = optimal_throughput / optimal_concurrency
                calculation_type = "direct"
            else:
                # 不满足条件，需要进一步计算满足 吞吐量/并发数 ≥ 10 的并发数
                optimal_concurrency = None
                optimal_throughput = None
                optimal_per_user = None
                calculation_type = "optimized"
                
                # 在所有延时严格小于要求的数据点中寻找最优解
                valid_points_strict = []
                for i, (conc, delay) in enumerate(zip(concurrencies, first_token_delays)):
                    if delay < max_first_token_delay:  # 严格小于
                        valid_points_strict.append((conc, delay, throughputs[i]))
                
                # 方法1：遍历所有满足延时要求的配置
                best_concurrency = 0
                for conc, delay, throughput in valid_points_strict:
                    per_user = throughput / conc
                    #if per_user >= 10.0 and conc > best_concurrency and delay < max_first_token_delay:  # 确保延时严格小于
                    if per_user >= user_throughput and conc > best_concurrency and delay < max_first_token_delay:
                        best_concurrency = conc
                        optimal_concurrency = conc
                        optimal_throughput = throughput
                        optimal_per_user = per_user
                
                # 方法2：如果还没找到合适的，通过插值方法寻找最优解
                if optimal_concurrency is None and len(valid_points_strict) >= 2:
                    # 在满足延时要求的数据点中，寻找单用户吞吐量刚好等于10的点
                    valid_points_strict.sort(key=lambda x: x[0])  # 按并发数排序
                    
                    for i in range(len(valid_points_strict) - 1):
                        conc1, delay1, throughput1 = valid_points_strict[i]
                        conc2, delay2, throughput2 = valid_points_strict[i + 1]
                        
                        per_user1 = throughput1 / conc1
                        per_user2 = throughput2 / conc2
                        
                        # 如果一个点满足单用户吞吐量要求，另一个不满足，在之间插值
                        #if per_user1 >= 10.0 > per_user2:
                        if per_user1 >= user_throughput > per_user2:
                            # 在这两点之间寻找单用户吞吐量刚好为10的点
                            # 使用数值方法求解 throughput(c) / c = 10
                            for ratio in [i / 100.0 for i in range(101)]:
                                test_conc = conc1 + (conc2 - conc1) * ratio
                                test_throughput = throughput1 + (throughput2 - throughput1) * ratio
                                test_delay = delay1 + (delay2 - delay1) * ratio
                                
                                if test_delay < max_first_token_delay:  # 确保延时严格小于
                                    test_per_user = test_throughput / test_conc
                                    #if abs(test_per_user - 10.0) < 0.05:  # 精度0.05
                                    if abs(test_per_user - user_throughput) < 0.05:
                                        if optimal_concurrency is None or test_conc > optimal_concurrency:
                                            optimal_concurrency = test_conc
                                            optimal_throughput = test_throughput
                                            optimal_per_user = test_per_user
                            break
                
                if optimal_concurrency is None:
                    return f"警告: 无法找到同时满足延时和单用户吞吐量要求(≥{user_throughput} tokens/s)的配置！", None,0
            
            # 生成详细报告
            interpolation_info = "（通过线性插值计算）" if interpolation_used else "（使用数据点）"
            
            report = (
                f'## 📊 <span style="color:{COLOR_PALETTE["标注"]}">性能分析报告</span>\n\n'
                f'**配置参数**\n'
                f'- 🖥️ GPU型号: `{gpu_model}`\n'
                f'- 🧠 模型名称: `{model_name}`\n' 
                f'- 🔢 卡数: `{card_count}`\n'
                f'- 📏 输入长度: `{input_length:,}` tokens\n'
                f'- ⏱️ 延时要求: `< {max_first_token_delay}s`\n\n'
                
                f'**最优配置** {interpolation_info}\n'
                f'- 🚀 <span style="color:{COLOR_PALETTE["错误"]}">最大并发数</span>: `{int(optimal_concurrency):,}` 并发\n'
                f'- 📈 <span style="color:{COLOR_PALETTE["趋势线"]}">系统总吞吐量</span>: `{optimal_throughput:,.2f}` tokens/s\n'
                f'- 👤 <span style="color:{COLOR_PALETTE["趋势线"]}">单用户吞吐量</span>: `{optimal_per_user:.2f}` tokens/s\n\n'
                        
            )
            
            
        
            all_valid_points = []
            
            # 添加原始数据点
            for i, (conc, delay) in enumerate(zip(concurrencies, first_token_delays)):
                if delay < max_first_token_delay:  # 严格小于
                    all_valid_points.append((conc, delay, throughputs[i], "数据点"))
            
            # 添加插值计算的点
            if interpolation_used:
                # 计算插值点的延时
                interpolated_delay = max_first_token_delay - 0.001
                all_valid_points.append((max_acceptable_concurrency, interpolated_delay, estimated_throughput, "插值点"))
            
            # 按并发数排序显示
            all_valid_points.sort(key=lambda x: x[0])
            for conc, delay, throughput, point_type in all_valid_points:
                per_user = throughput / conc
                #status = "✅" if per_user >= 10.0 else "⚠️"
                status = "✅" if per_user >= user_throughput else "⚠️"
                delay_status = "✅" if delay < max_first_token_delay else "❌"
                
                # report += (
                #     f"  {status}{delay_status} 并发 {conc:>5,.0f}: "
                #     f"{throughput:>8,.0f} tokens/s | "
                #     f"延时 {delay:>5.3f}s | "
                #     f"单用户 {per_user:>5.1f} tokens/s\n"
                # )
            # 生成优化后的图表
            chart = self.create_optimized_performance_chart(
                gpu_model, model_name, card_count, input_length, 
                max_first_token_delay, optimal_concurrency, optimal_throughput, user_throughput
            )
            
            return (
                report, 
                chart,
                int(optimal_concurrency)  # 新增结构化数据但保持向下兼容
            )
        
        except Exception as e:
            return f"计算出错: {str(e)}", None , None

    def calculate_cost_optimization(self,max_concurrency: int, selected_model: str, 
                               context_length: int, max_delay: float) -> List[Dict]:
        current_performance_data=self.current_performance_data
        
        solutions = []
        
        # 遍历所有GPU型号
        for gpu_model, model_data in current_performance_data.items():
            if selected_model not in model_data:
                continue
                
            # 遍历该型号下的卡数配置
            for card_count, card_data in model_data[selected_model].items():
                if context_length not in card_data:
                    continue
                
                # 获取结构化返回值
                report, chart, max_conc_per_node = self.calculate_max_concurrency_optimized(
                    gpu_model, selected_model, card_count, context_length, max_delay
                )
                
                # 错误处理逻辑
                if not isinstance(max_conc_per_node, int) or max_conc_per_node <= 0:
                    continue
                    
                # 计算所需节点数（添加最小节点数限制）
                nodes_required = max(1, np.ceil(max_concurrency / max_conc_per_node))
                
                # 计算总卡数和成本（添加单价校验）
                if gpu_model not in GPU_PRICES:
                    continue
                    
                total_cards = nodes_required * card_count
                total_price = total_cards * GPU_PRICES[gpu_model]["price"]
                
                solutions.append({
                    "gpu_model": gpu_model,
                    "card_count_per_node": card_count,
                    "nodes_required": int(nodes_required),
                    "total_cards": int(total_cards),
                    "total_price": total_price,
                    "max_conc_per_node": max_conc_per_node
                })
        
        # 添加排序逻辑确保有效结果
        valid_solutions = [s for s in solutions if s['total_price'] > 0]
        return sorted(valid_solutions, key=lambda x: x["total_price"])[:3]

