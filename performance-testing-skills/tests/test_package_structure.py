from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_expected_package_files_exist():
    expected = [
        "README.md",
        "skills/automated-performance-testing/SKILL.md",
        "skills/manual-interface-performance-testing/SKILL.md",
        "scripts/perf_auto.py",
        "scripts/perf_manual.py",
        "scripts/run_auto.sh",
        "scripts/run_manual.sh",
        "scripts/lib/__init__.py",
        "scripts/lib/config.py",
        "scripts/lib/datasets.py",
        "scripts/lib/clients.py",
        "scripts/lib/metrics.py",
        "scripts/lib/reporters.py",
        "configs/openai_chat.json",
        "configs/openai_completion.json",
        "configs/openai_asr.json",
        "configs/generic_http.json",
        "datasets/text_prompts.example.jsonl",
        "datasets/audio_manifest.example.jsonl",
        "reports/.gitkeep",
    ]
    missing = [path for path in expected if not (ROOT / path).exists()]
    assert missing == []


def test_skill_frontmatter_is_valid():
    skills = [
        ROOT / "skills/automated-performance-testing/SKILL.md",
        ROOT / "skills/manual-interface-performance-testing/SKILL.md",
    ]
    for path in skills:
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert "\nname: " in text
        assert "\ndescription: " in text
        assert "\n---\n" in text[4:]
