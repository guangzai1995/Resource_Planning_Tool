"""Built-in offline chat-style prompts for MTP/spec decode benchmarks."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _TopicSeed:
    topic: str
    system: str
    evidence: str
    instruction: str


@dataclass(frozen=True)
class _Candidate:
    prompt: str
    prompt_len: int


_TOPIC_SEEDS = (
    _TopicSeed(
        topic="中文长文摘要与关键结论提取",
        system="你是严谨的中文技术文档分析助手。",
        evidence=(
            "项目 背景 资源 调度 集群 容量 配额 峰值 负载 业务 诉求 "
            "风险 约束 上下游 依赖 交付 时间线 指标 结论 取舍 建议 "
            "稳定性 成本 利用率 排队 延迟 吞吐 优先级 回滚 观测 告警"
        ),
        instruction="请总结关键事实、列出风险、给出三条可执行建议",
    ),
    _TopicSeed(
        topic="多轮技术问答和需求澄清",
        system="你是负责需求澄清的后端架构师。",
        evidence=(
            "用户 希望 自动化 测试 支持 多模型 多后端 多配置 矩阵 "
            "需要 保留 现有 字段 兼容 历史 结果 需要 错误 信息 清晰 "
            "需要 区分 默认 行为 新行为 灰度 发布 文档 示例 验收"
        ),
        instruction="请先复述需求边界，再提出需要确认的问题，并给出默认实现方案",
    ),
    _TopicSeed(
        topic="Python 代码阅读与问题定位",
        system="你是资深 Python 测试框架维护者。",
        evidence=(
            "函数 参数 headers base_url session response status text async "
            "上下文 管理器 异常 捕获 返回 None 指标 解析 Prometheus "
            "字典 字段 默认值 CSV 表头 单元 测试 回归 覆盖 兼容"
        ),
        instruction="请定位潜在缺陷，解释触发条件，并给出最小补丁思路",
    ),
    _TopicSeed(
        topic="服务日志分析和根因判断",
        system="你是线上推理服务的故障分析专家。",
        evidence=(
            "日志 时间 请求 路径 状态码 401 Authorization Bearer metrics "
            "chat completions 成功 200 readiness models probe 容器 网络 "
            "反向 代理 鉴权 头部 缺失 重试 超时 指标 采集 差分"
        ),
        instruction="请根据日志归纳症状、根因证据、排除项和修复验证步骤",
    ),
    _TopicSeed(
        topic="JSON 配置审查和字段解释",
        system="你是自动化 benchmark 配置审查助手。",
        evidence=(
            "bench_profiles dataset name builtin_mtp_chat length_policy bucket "
            "input_len_tolerance on_bucket_shortage sampling input_lens output_lens "
            "parallel_nums epochs cross_product backend tokenizer api_key serve_profiles args"
        ),
        instruction="请检查字段语义是否一致，指出冲突配置，并输出规范化配置片段",
    ),
    _TopicSeed(
        topic="数学推理与步骤化演算",
        system="你是关注可验证步骤的数学推理助手。",
        evidence=(
            "吞吐 请求数 并发 轮数 平均 输入 输出 token duration TTFT TPOT "
            "缓存 命中率 接受率 草稿 token 接受 token 比例 百分比 阈值 "
            "上下界 bucket tolerance target lower upper 计算 校验"
        ),
        instruction="请逐步计算关键指标，保留公式，并解释边界条件",
    ),
    _TopicSeed(
        topic="中英混合 API 使用说明",
        system="You are a bilingual API integration assistant.",
        evidence=(
            "OpenAI compatible endpoint base_url model served_model_name api_key "
            "headers Authorization Bearer timeout concurrency max_tokens prompt "
            "dataset random builtin_mtp_chat tokenizer chat_template response usage"
        ),
        instruction="Please produce a concise bilingual guide with pitfalls and validation commands",
    ),
    _TopicSeed(
        topic="测试报告总结与风险分级",
        system="你是负责质量门禁的测试负责人。",
        evidence=(
            "测试 结果 passed failed skipped warning pytest baseline regression "
            "定向 用例 全量 验证 覆盖 风险 等级 必须 修复 建议 修改 "
            "残余 风险 性能 指标 接受率 稳定性 兼容性 输出 报告"
        ),
        instruction="请按严重性排序问题，给出验收结论和剩余风险",
    ),
)


def _repeat_to_budget(seed_text: str, target_words: int) -> str:
    words = seed_text.split()
    if not words:
        return seed_text
    chunks: list[str] = []
    while len(chunks) < target_words:
        chunks.extend(words)
    return " ".join(chunks[:target_words])


def _render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return rendered if isinstance(rendered, str) else str(rendered)
        except TypeError:
            rendered = tokenizer.apply_chat_template(messages)
            return rendered if isinstance(rendered, str) else str(rendered)
    return "\n".join(f"{item['role']}: {item['content']}" for item in messages)


def _encoded_length(encoded: Any) -> int:
    if isinstance(encoded, dict) and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif hasattr(encoded, "ids"):
        encoded = encoded.ids
    if isinstance(encoded, list) and encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return len(encoded)


def _count_tokens(tokenizer: Any, messages: list[dict[str, str]], prompt: str) -> int:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            return _encoded_length(encoded)
        except (TypeError, ValueError):
            pass
    if hasattr(tokenizer, "encode"):
        return _encoded_length(tokenizer.encode(prompt))
    encoded = tokenizer(prompt, add_special_tokens=False)
    return _encoded_length(encoded)


def _make_messages(seed: _TopicSeed, target_words: int, variant: int) -> list[dict[str, str]]:
    evidence = _repeat_to_budget(seed.evidence, target_words)
    user = (
        f"主题: {seed.topic}\n"
        f"样本编号: {variant}\n"
        f"上下文材料:\n{evidence}\n\n"
        f"任务: {seed.instruction}。"
    )
    return [
        {"role": "system", "content": seed.system},
        {"role": "user", "content": user},
    ]


def _build_candidate(
    tokenizer: Any,
    seed: _TopicSeed,
    target_len: int,
    lower: int,
    upper: int,
    variant: int,
) -> _Candidate | None:
    word_budget = max(8, target_len - 24)
    best: _Candidate | None = None
    for _ in range(8):
        messages = _make_messages(seed, word_budget, variant)
        prompt = _render_prompt(tokenizer, messages)
        prompt_len = _count_tokens(tokenizer, messages, prompt)
        best = _Candidate(prompt=prompt, prompt_len=prompt_len)
        if lower <= prompt_len <= upper:
            return best
        if prompt_len <= 0:
            return None
        scaled_budget = int(word_budget * target_len / prompt_len)
        if scaled_budget == word_budget:
            scaled_budget += 1 if prompt_len < lower else -1
        word_budget = max(8, scaled_budget)
    if best is not None and lower <= best.prompt_len <= upper:
        return best
    return None


def _positive_int_from_args(args: Any, primary: str, fallback: str) -> int:
    value = getattr(args, primary, None) or getattr(args, fallback, None)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"builtin_mtp_chat requires positive {primary}")
    return value


def _bucket_bounds(args: Any, target_len: int) -> tuple[int, int]:
    policy = getattr(args, "dataset_length_policy", "exact")
    if policy not in {"exact", "bucket"}:
        raise ValueError("builtin_mtp_chat supports dataset_length_policy exact or bucket")
    tolerance = 0.0
    if policy == "bucket":
        tolerance = float(getattr(args, "dataset_input_len_tolerance", 0.2))
    if tolerance < 0 or tolerance >= 1:
        raise ValueError("dataset_input_len_tolerance must be >= 0 and < 1")
    lower = max(1, int(target_len * (1 - tolerance)))
    upper = max(lower, int(target_len * (1 + tolerance)))
    return lower, upper


def _build_candidates(
    args: Any,
    tokenizer: Any,
    target_len: int,
    lower: int,
    upper: int,
) -> list[_Candidate]:
    num_prompts = _positive_int_from_args(args, "num_prompts", "num_prompts")
    target_candidates = min(max(num_prompts, len(_TOPIC_SEEDS) * 4), 512)
    candidates: list[_Candidate] = []
    variant = 0
    while len(candidates) < target_candidates and variant < target_candidates * 3:
        seed = _TOPIC_SEEDS[variant % len(_TOPIC_SEEDS)]
        candidate = _build_candidate(
            tokenizer=tokenizer,
            seed=seed,
            target_len=target_len,
            lower=lower,
            upper=upper,
            variant=variant,
        )
        if candidate is not None:
            candidates.append(candidate)
        variant += 1
    return candidates


def build_requests(args: Any, tokenizer: Any, sample_request_cls: type) -> list[Any]:
    if tokenizer is None:
        raise ValueError("builtin_mtp_chat requires --tokenizer")

    target_len = _positive_int_from_args(args, "input_len", "random_input_len")
    output_len = _positive_int_from_args(args, "output_len", "random_output_len")
    num_prompts = _positive_int_from_args(args, "num_prompts", "num_prompts")
    lower, upper = _bucket_bounds(args, target_len)
    candidates = _build_candidates(args, tokenizer, target_len, lower, upper)
    if not candidates:
        raise ValueError(
            f"builtin_mtp_chat has no prompts in token bucket [{lower}, {upper}] "
            f"for target input_len={target_len}; increase input_len_tolerance"
        )

    sampling = getattr(args, "dataset_sampling", "shuffle")
    if sampling not in {"shuffle", "round_robin"}:
        raise ValueError("builtin_mtp_chat supports dataset_sampling shuffle or round_robin")
    if sampling == "shuffle":
        rng = random.Random(int(getattr(args, "seed", 0)))
        rng.shuffle(candidates)

    selected = [candidates[index % len(candidates)] for index in range(num_prompts)]
    return [
        sample_request_cls(
            prompt=item.prompt,
            prompt_len=item.prompt_len,
            expected_output_len=output_len,
        )
        for item in selected
    ]
