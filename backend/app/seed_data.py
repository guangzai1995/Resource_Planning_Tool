"""
初始 GPU 和模型数据（来自 资源规划工具.xlsx 测试数据）
"""

INITIAL_GPUS = [
    {
        "name": "H200",
        "vendor": "NVIDIA",
        "memory_gb": 141.0,
        "memory_bandwidth_gbps": 4800.0,
        "tflops_bf16": 989.0,
        "price_per_hour": 60.0,
        "notes": "NVIDIA H200 SXM5 141GB HBM3e",
    },
    {
        "name": "H20",
        "vendor": "NVIDIA",
        "memory_gb": 96.0,
        "memory_bandwidth_gbps": 4000.0,
        "tflops_bf16": 296.0,
        "price_per_hour": 30.0,
        "notes": "NVIDIA H20 96GB",
    },
    {
        "name": "P800",
        "vendor": "Huawei",
        "memory_gb": 64.0,
        "memory_bandwidth_gbps": 2000.0,
        "tflops_bf16": 280.0,
        "price_per_hour": 20.0,
        "notes": "华为昇腾 910B 系列（估算参数，可在设置页面更新）",
    },
]

INITIAL_MODELS = [
    {
        "name": "4B",
        "parameter_b": 4.0,
        "model_type": "dense",
        "is_moe": 0,
        "num_layers": 36,
        "hidden_size": 2560,
        "num_kv_heads": 8,
        "head_size": 128,
        "notes": "Qwen3-4B",
    },
    {
        "name": "7B",
        "parameter_b": 7.0,
        "model_type": "dense",
        "is_moe": 0,
        "num_layers": 28,
        "hidden_size": 3584,
        "num_kv_heads": 4,
        "head_size": 128,
        "notes": "Qwen2.5-7B-Instruct",
    },
    {
        "name": "14B",
        "parameter_b": 14.0,
        "model_type": "dense",
        "is_moe": 0,
        "num_layers": 40,
        "hidden_size": 5120,
        "num_kv_heads": 8,
        "head_size": 128,
        "notes": "Qwen2.5-14B-Instruct",
    },
    {
        "name": "32B",
        "parameter_b": 32.0,
        "model_type": "dense",
        "is_moe": 0,
        "num_layers": 64,
        "hidden_size": 5120,
        "num_kv_heads": 8,
        "head_size": 128,
        "notes": "Qwen2.5-32B-Instruct",
    },
    {
        "name": "72B",
        "parameter_b": 72.0,
        "model_type": "dense",
        "is_moe": 0,
        "num_layers": 80,
        "hidden_size": 8192,
        "num_kv_heads": 8,
        "head_size": 128,
        "notes": "Qwen2.5-72B-Instruct",
    },
    {
        "name": "235B-A22B",
        "parameter_b": 235.0,
        "model_type": "moe",
        "is_moe": 1,
        "num_layers": 94,
        "hidden_size": 7168,
        "num_kv_heads": 4,
        "head_size": 128,
        "notes": "Qwen3-235B-A22B (MoE)",
    },
    {
        "name": "671B",
        "parameter_b": 671.0,
        "model_type": "moe",
        "is_moe": 1,
        "num_layers": 61,
        "hidden_size": 7168,
        "num_kv_heads": 128,
        "head_size": 128,
        "notes": "DeepSeek-R1-671B (MoE)",
    },
]


def seed_initial_data(db):
    from app.models.gpu_spec import GpuSpec
    from app.models.model import Model

    for gpu_data in INITIAL_GPUS:
        if not db.query(GpuSpec).filter_by(name=gpu_data["name"]).first():
            db.add(GpuSpec(**gpu_data))

    for model_data in INITIAL_MODELS:
        if not db.query(Model).filter_by(name=model_data["name"]).first():
            db.add(Model(**model_data))

    db.commit()
