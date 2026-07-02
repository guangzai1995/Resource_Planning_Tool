import json
from pathlib import Path

from scripts.lib.config import load_config, package_root, resolve_package_path
from scripts.lib.datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_package_root_points_to_portable_package():
    assert package_root().name == "performance-testing-skills"


def test_load_config_expands_environment_variables(monkeypatch, tmp_path):
    monkeypatch.setenv("PERF_TEST_TOKEN", "secret-token")
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({
            "protocol": "openai_chat",
            "base_url": "http://127.0.0.1:8000/v1",
            "headers": {"Authorization": "Bearer ${PERF_TEST_TOKEN}"},
        }),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config["headers"]["Authorization"] == "Bearer secret-token"


def test_resolve_package_path_keeps_absolute_paths(tmp_path):
    assert resolve_package_path(str(tmp_path)) == tmp_path


def test_resolve_package_path_resolves_relative_to_package_root():
    resolved = resolve_package_path("datasets/text_prompts.example.jsonl")
    assert resolved == ROOT / "datasets/text_prompts.example.jsonl"


def test_load_text_prompts_dataset():
    rows = load_dataset({
        "type": "text_prompts",
        "path": "datasets/text_prompts.example.jsonl",
    })

    assert rows[0]["prompt"]
    assert isinstance(rows[0]["expected_output_len"], int)


def test_load_audio_manifest_dataset():
    rows = load_dataset({
        "type": "audio_manifest",
        "path": "datasets/audio_manifest.example.jsonl",
    })

    assert rows[0]["audio"].endswith(".wav")
    assert isinstance(rows[0]["duration_s"], float)
