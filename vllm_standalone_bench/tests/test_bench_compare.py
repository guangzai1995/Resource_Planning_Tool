import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace

import bench_compare as bc


CSV_HEADER = (
    "model,backend,input_len,output_len,parallel_num,epochs,num_prompts,n_success,"
    "n_failed,avg_input_tokens,avg_output_tokens,throughput_req_s,throughput_tok_s,"
    "ttft_mean_ms,ttft_p50_ms,ttft_p90_ms,ttft_p99_ms,tpot_mean_ms,tpot_p50_ms,"
    "tpot_p90_ms,tpot_p99_ms,e2el_mean_ms,e2el_p50_ms,e2el_p90_ms,e2el_p99_ms,duration_s"
)


def _write_result_csv(path: Path, parallel: int, ttft_p50: int, tput: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        CSV_HEADER + "\n"
        f"m,openai-chat,64,32,{parallel},1,1,1,0,64,32,1.0,{tput},10,{ttft_p50},"
        f"20,30,5,4,6,8,50,40,60,70,10\n",
        encoding="utf-8",
    )


def _fake_config():
    serve = [
        SimpleNamespace(name="vllm_bf16", engine="vllm", gpus="all", args=()),
        SimpleNamespace(name="sglang_bf16", engine="sglang", gpus="all", args=()),
    ]
    return SimpleNamespace(
        serve_profiles=serve,
        models=[SimpleNamespace(name="qwen")],
        bench_profiles=[SimpleNamespace(name="smoke")],
    )


def test_aggregate_aligns_two_engines_and_preserves_originals(tmp_path):
    config = _fake_config()
    run_dir = tmp_path / "run1"
    vllm_csv = run_dir / "qwen" / "vllm_bf16" / "smoke" / "result.csv"
    sglang_csv = run_dir / "qwen" / "sglang_bf16" / "smoke" / "result.csv"
    _write_result_csv(vllm_csv, parallel=1, ttft_p50=11, tput=100)
    _write_result_csv(sglang_csv, parallel=1, ttft_p50=22, tput=200)
    before_vllm = hashlib.sha256(vllm_csv.read_bytes()).hexdigest()
    before_sglang = hashlib.sha256(sglang_csv.read_bytes()).hexdigest()

    out = bc.aggregate_compare(config, run_dir)

    assert out == run_dir / "compare.csv"
    assert (run_dir / "compare.xlsx").exists()
    plots = list((run_dir / "plots").glob("*.png"))
    assert plots, "应至少生成一张图表"
    rows = list(csv.DictReader((run_dir / "compare.csv").open(encoding="utf-8-sig")))
    assert len(rows) == 1
    row = rows[0]
    assert row["vllm__throughput_tok_s"] == "100"
    assert row["sglang__throughput_tok_s"] == "200"
    assert row["vllm__ttft_p50_ms"] == "11"
    assert row["sglang__ttft_p50_ms"] == "22"
    # 原始 result.csv 必须未被修改
    assert hashlib.sha256(vllm_csv.read_bytes()).hexdigest() == before_vllm
    assert hashlib.sha256(sglang_csv.read_bytes()).hexdigest() == before_sglang


def test_aggregate_missing_engine_fills_na(tmp_path):
    config = _fake_config()
    run_dir = tmp_path / "run1"
    _write_result_csv(run_dir / "qwen" / "vllm_bf16" / "smoke" / "result.csv",
                      parallel=1, ttft_p50=11, tput=100)

    out = bc.aggregate_compare(config, run_dir)

    rows = list(csv.DictReader(out.open(encoding="utf-8-sig")))
    assert rows[0]["vllm__throughput_tok_s"] == "100"
    assert rows[0]["sglang__throughput_tok_s"] == "N/A"


def test_aggregate_no_results_returns_none(tmp_path):
    config = _fake_config()

    assert bc.aggregate_compare(config, tmp_path / "empty") is None
    assert not (tmp_path / "empty" / "compare.csv").exists()
