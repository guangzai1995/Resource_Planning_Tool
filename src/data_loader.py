from pathlib import Path
import re
import pandas as pd
import gradio as gr

class DataLoader:
    def __init__(self, data_dir: str="./data"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist.")
        self.performance_data= self.load_performance_data(self.data_dir)
    
    def load_performance_data(self, data_dir: Path):
        """从data目录下的CSV文件加载性能数据"""
        performance_data = {}
        data_path = data_dir
        
        # 遍历GPU型号目录
        for gpu_dir in data_path.iterdir():
            if gpu_dir.is_dir():
                gpu_model = gpu_dir.name
                
                # 遍历模型目录
                for model_dir in gpu_dir.iterdir():
                    if model_dir.is_dir():
                        model_name = model_dir.name
                        
                        # 遍历CSV文件
                        # 支持大小写后缀(有些目录中文件为大写 .CSV)
                        csv_files = list(model_dir.glob("*.csv")) + list(model_dir.glob("*.CSV"))
                        for csv_file in csv_files:
                            # 从文件名提取卡数
                            card_match = re.search(r'(\d+)', csv_file.stem)
                            if card_match:
                                card_count = int(card_match.group(1))
                                
                                try:
                                    # 读取CSV文件
                                    df = pd.read_csv(csv_file)
                                    
                                    # 构建层级数据结构
                                    if gpu_model not in performance_data:
                                        performance_data[gpu_model] = {}
                                    if model_name not in performance_data[gpu_model]:
                                        performance_data[gpu_model][model_name] = {}
                                    if card_count not in performance_data[gpu_model][model_name]:
                                        performance_data[gpu_model][model_name][card_count] = {}
                                    
                                    # 处理CSV数据
                                    for _, row in df.iterrows():
                                        input_length = int(row.get('输入长度', 0))
                                        concurrency = int(row.get('并发数', 0))
                                        throughput = float(row.get('输出tokens总吞吐', 0))
                                        first_token = float(row.get('平均首tokens时延（ms）', 0))
                                        incremental_delay = float(row.get('平均增量时延（ms）', 0))

                                        if input_length not in performance_data[gpu_model][model_name][card_count]:
                                            performance_data[gpu_model][model_name][card_count][input_length] = {}
                                        
                                        performance_data[gpu_model][model_name][card_count][input_length][concurrency] = {
                                            "throughput": throughput,
                                            "first_token": first_token,
                                            "incremental_delay": incremental_delay
                                        }
                                        
                                except Exception as e:
                                    print(f"读取文件 {csv_file} 时出错: {e}")
        
        return performance_data
    
    def update_model_choices(self,gpu_model):
        """根据选择的GPU更新模型选择"""
        
        if not gpu_model or gpu_model not in self.performance_data:
            return gr.update(choices=[], value=None)
        
        models = sorted(self.performance_data[gpu_model].keys())
        return gr.update(choices=models, value=models[0] if models else None)
    
    def update_card_choices(self,gpu_model, model_name):
        """根据选择的GPU和模型更新卡数选择"""
        if (not gpu_model or not model_name or 
            gpu_model not in self.performance_data or 
            model_name not in self.performance_data[gpu_model]):
            return gr.update(choices=[], value=None)
        
        cards = sorted(self.performance_data[gpu_model][model_name].keys())
        return gr.update(choices=cards, value=cards[0] if cards else None)
    
    def update_input_length_choices(self,gpu_model, model_name, card_count):
        """根据选择的GPU、模型和卡数更新输入长度选择"""
        if (not gpu_model or not model_name or not card_count or
            gpu_model not in self.performance_data or 
            model_name not in self.performance_data[gpu_model] or
            card_count not in self.performance_data[gpu_model][model_name]):
            return gr.update(choices=[], value=None)
        
        input_lengths = sorted(self.performance_data[gpu_model][model_name][card_count].keys())
        return gr.update(choices=input_lengths, value=input_lengths[0] if input_lengths else None)


    def initialize_cascading_updates(self,gpu_model):
        """初始化级联更新"""
        # 更新模型选择
        model_update = self.update_model_choices(gpu_model)
        models = model_update.get('choices', [])
        selected_model = model_update.get('value', None)
        
        # 更新卡数选择
        card_update = self.update_card_choices(gpu_model, selected_model)
        cards = card_update.get('choices', [])
        selected_card = card_update.get('value', None)
        
        # 更新输入长度选择
        input_update = self.update_input_length_choices(gpu_model, selected_model, selected_card)
        
        return model_update, card_update, input_update

    def refresh_data(self):
        """刷新数据并更新界面"""
        # 重新加载性能数据
        self.performance_data = self.load_performance_data(self.data_dir)
        
        gpu_models = sorted(self.performance_data.keys())
        if not gpu_models:
            return (
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=None),
                "数据已刷新！未找到GPU型号数据。"
            )
        
        # 使用第一个GPU型号初始化级联更新
        model_update, card_update, input_update = self.initialize_cascading_updates(gpu_models[0])
        
        return (
            gr.update(choices=gpu_models, value=gpu_models[0]),
            model_update,
            card_update,
            input_update,
            f"数据已刷新！共加载 {len(gpu_models)} 个GPU型号。"
        )
    
    def get_available_options(self):

    
        """获取可用的选项"""
        if not self.performance_data:
            return [], [], [], []
        
        gpu_models = sorted(self.performance_data.keys())
        
        # 获取所有模型名称
        model_names = set()
        card_counts = set()
        input_lengths = set()

        for gpu_data in self.performance_data.values():
            for model_name, model_data in gpu_data.items():
                model_names.add(model_name)
                for card_count, card_data in model_data.items():
                    card_counts.add(card_count)
                    for input_len in card_data.keys():
                        input_lengths.add(input_len)
        
        return gpu_models, sorted(model_names), sorted(card_counts), sorted(input_lengths)