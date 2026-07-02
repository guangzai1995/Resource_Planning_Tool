from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "manual_asr_smoke.sh"


def test_manual_asr_smoke_script_is_runnable():
    assert SCRIPT.exists()
    assert SCRIPT.stat().st_mode & 0o111

    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


def test_manual_asr_smoke_script_defaults_match_known_endpoint():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "http://10.86.0.32:13001/v1/audio/transcriptions" in text
    assert "Qwen3-ASR-1_7B" in text
    assert "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav" in text
    assert "language=${LANGUAGE}" in text


def test_manual_asr_smoke_script_retries_transient_connection_failures():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "CURL_RETRY" in text
    assert "--retry-connrefused" in text
