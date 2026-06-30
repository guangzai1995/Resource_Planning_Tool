from pathlib import Path

import pandas as pd
import pytest

from scripts.estimate_deepseek_v4_flash_ascend import (
    DEFAULT_GPU_ASSUMPTIONS,
    DEFAULT_MODEL_ASSUMPTION,
    DEFAULT_RUNTIME_SCENARIOS,
    SOURCE_COLUMNS,
    BaselineRow,
    build_estimates,
    estimate_row,
    load_p800_baseline,
    required_memory_per_card_gb,
    summarize_910b_reference,
)


def test_required_memory_per_card_changes_with_weight_precision():
    bf16 = required_memory_per_card_gb(
        model=DEFAULT_MODEL_ASSUMPTION.with_updates(weight_bytes=2.0),
        gpu_count=8,
        input_tokens=512,
        output_tokens=1024,
        concurrency=5,
        framework_overhead_gb=3.0,
    )
    int8 = required_memory_per_card_gb(
        model=DEFAULT_MODEL_ASSUMPTION.with_updates(weight_bytes=1.0),
        gpu_count=8,
        input_tokens=512,
        output_tokens=1024,
        concurrency=5,
        framework_overhead_gb=3.0,
    )
    assert bf16 > int8
    assert int8 > 3.0


def test_910c_compute_scale_uses_target_hardware_ratio():
    baseline = BaselineRow(
        excel_row=2,
        input_tokens=512,
        output_tokens=1024,
        concurrency=5,
        throughput_tokens_s=100.0,
        ttft_p90_ms=400.0,
        ttft_p99_ms=500.0,
        ttft_max_ms=550.0,
        ttft_mean_ms=300.0,
        decode_latency_p90_ms=40.0,
        decode_latency_p99_ms=45.0,
        decode_latency_max_ms=60.0,
        decode_latency_mean_ms=35.0,
    )
    result = estimate_row(
        baseline=baseline,
        target_gpu=DEFAULT_GPU_ASSUMPTIONS["910C"],
        base_gpu=DEFAULT_GPU_ASSUMPTIONS["P800"],
        model=DEFAULT_MODEL_ASSUMPTION.with_updates(weight_bytes=1.0),
        runtime=DEFAULT_RUNTIME_SCENARIOS["base"],
    )
    assert result["target_gpu"] == "910C"
    assert result["gpu_count"] == 8
    assert result["compute_scale"] == pytest.approx(780 / 280, rel=1e-6)
    assert result["decode_scale"] == pytest.approx(
        min(result["compute_scale"], result["bandwidth_scale"]),
        rel=1e-6,
    )
    assert result["输出tokens总吞吐"] == pytest.approx(100.0 * result["decode_scale"])
    assert 0.0 <= result["confidence"] <= 1.0


def write_baseline_workbook(path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "输入长度": 512,
                "输出长度": 1024,
                "并发数": 5,
                "输出tokens总吞吐": 117.0,
                "首tokens时延TP90（ms）": 476.0,
                "首tokens时延TP99（ms）": 500.0,
                "最大首tokens时延（ms）": 501.0,
                "平均首tokens时延（ms）": 381.0,
                "增量时延TP90（ms）": 43.0,
                "增量时延TP99（ms）": 44.0,
                "最大增量时延（ms）": 46.0,
                "平均增量时延（ms）": 42.0,
            },
            {
                "输入长度": 1024,
                "输出长度": 1024,
                "并发数": 10,
                "输出tokens总吞吐": 150.0,
                "首tokens时延TP90（ms）": 600.0,
                "首tokens时延TP99（ms）": 650.0,
                "最大首tokens时延（ms）": 700.0,
                "平均首tokens时延（ms）": 500.0,
                "增量时延TP90（ms）": 55.0,
                "增量时延TP99（ms）": 58.0,
                "最大增量时延（ms）": 70.0,
                "平均增量时延（ms）": 50.0,
            },
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="671B-P800-8测试数据", index=False)


def test_load_p800_baseline_preserves_excel_row_numbers(tmp_path):
    workbook = tmp_path / "资源规划工具.xlsx"
    write_baseline_workbook(workbook)
    rows = load_p800_baseline(workbook)
    assert len(rows) == 2
    assert rows[0].excel_row == 2
    assert rows[1].excel_row == 3
    assert rows[0].throughput_tokens_s == 117.0


def test_build_estimates_creates_910b_and_910c_rows(tmp_path):
    workbook = tmp_path / "资源规划工具.xlsx"
    write_baseline_workbook(workbook)
    baseline_rows = load_p800_baseline(workbook)
    estimates = build_estimates(baseline_rows)
    assert set(estimates["target_gpu"]) == {"910B", "910C"}
    assert len(estimates) == 4
    assert list(estimates.columns[: len(SOURCE_COLUMNS)]) == SOURCE_COLUMNS
    assert {"base_row", "compute_scale", "bandwidth_scale", "decode_scale", "confidence"}.issubset(estimates.columns)


def write_reference_csv(path: Path, throughput: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "输入长度,输出长度,并发数,输出tokens总吞吐,首tokens时延TP90（ms）,首tokens时延TP99（ms）,最大首tokens时延（ms）,平均首tokens时延（ms）,增量时延TP90（ms）,增量时延TP99（ms）,最大增量时延（ms）,平均增量时延（ms）",
                f"2048,1200,1,{throughput},320,330,340,310,30,35,55,29",
                f"2048,1200,4,{throughput * 2},1100,1110,1120,1000,32,38,58,30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_910b_reference_summary_does_not_change_estimates(tmp_path):
    workbook = tmp_path / "资源规划工具.xlsx"
    write_baseline_workbook(workbook)
    baseline_rows = load_p800_baseline(workbook)
    before = build_estimates(baseline_rows)

    write_reference_csv(tmp_path / "data" / "910B1" / "72B-AWQ" / "8.csv", 10.0)
    first_summary = summarize_910b_reference(tmp_path / "data")
    write_reference_csv(tmp_path / "data" / "910B1" / "72B-AWQ" / "8.csv", 9999.0)
    second_summary = summarize_910b_reference(tmp_path / "data")
    after = build_estimates(baseline_rows)

    assert first_summary.loc[0, "mean_throughput_tokens_s"] != second_summary.loc[0, "mean_throughput_tokens_s"]
    pd.testing.assert_frame_equal(before, after)
