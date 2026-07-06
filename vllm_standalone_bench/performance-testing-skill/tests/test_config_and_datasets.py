import importlib.util
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, PACKAGE_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_loads_example_chat_config_with_resolved_dataset_path():
    config_module = load_module("skill_config", "scripts/lib/config.py")

    config = config_module.load_config(PACKAGE_ROOT / "configs/openai_chat.json")

    assert config.run.name == "openai-chat-smoke"
    assert config.target.type == "openai-chat"
    assert config.target.endpoint == "/v1/chat/completions"
    assert config.dataset.path == PACKAGE_ROOT / "datasets/text_prompts.example.jsonl"
    assert config.request_count == 4
    assert config.concurrency == 2


def test_rejects_invalid_config(tmp_path):
    config_module = load_module("skill_config_invalid", "scripts/lib/config.py")
    bad_config = tmp_path / "bad.json"
    bad_config.write_text('{"target": {"type": "openai-chat"}}', encoding="utf-8")

    try:
        config_module.load_config(bad_config)
    except config_module.ConfigError as exc:
        assert "run" in str(exc)
    else:
        raise AssertionError("load_config should reject missing run section")


def test_loads_text_prompts_and_audio_manifest():
    dataset_module = load_module("skill_datasets", "scripts/lib/datasets.py")

    prompts = dataset_module.load_dataset(PACKAGE_ROOT / "datasets/text_prompts.example.jsonl")
    audio_items = dataset_module.load_dataset(PACKAGE_ROOT / "datasets/audio_manifest.example.jsonl")

    assert prompts[0]["prompt"]
    assert prompts[0]["metadata"]["case"] == "short"
    assert audio_items[0]["audio_path"].endswith(".wav")
    assert audio_items[0]["prompt"]


def test_cycles_samples_to_requested_count():
    dataset_module = load_module("skill_datasets_cycle", "scripts/lib/datasets.py")

    samples = [{"prompt": "a"}, {"prompt": "b"}]
    cycled = dataset_module.expand_samples(samples, 5)

    assert [item["prompt"] for item in cycled] == ["a", "b", "a", "b", "a"]
