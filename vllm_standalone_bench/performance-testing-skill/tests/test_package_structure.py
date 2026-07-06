from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_skill_package_layout():
    expected_files = [
        "SKILL.md",
        "scripts/perf_auto.py",
        "scripts/perf_manual.py",
        "scripts/run_auto.sh",
        "scripts/run_manual.sh",
        "scripts/lib/__init__.py",
        "scripts/lib/clients.py",
        "scripts/lib/config.py",
        "scripts/lib/datasets.py",
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

    for relative_path in expected_files:
        assert (PACKAGE_ROOT / relative_path).is_file(), relative_path

    assert not (PACKAGE_ROOT / "skills").exists()


def test_skill_frontmatter_is_valid_and_actionable():
    content = (PACKAGE_ROOT / "SKILL.md").read_text(encoding="utf-8")

    assert content.startswith("---\n")
    assert "name: generic-performance-testing" in content
    assert "description: Use when" in content
    assert "## 工作流程" in content
    assert "scripts/run_auto.sh" in content


def test_package_does_not_depend_on_parent_benchmark_project():
    forbidden = ["from vllm_standalone_bench", "import vllm_standalone_bench"]
    checked_files = [
        *PACKAGE_ROOT.glob("SKILL.md"),
        *PACKAGE_ROOT.glob("scripts/**/*.py"),
        *PACKAGE_ROOT.glob("scripts/*.sh"),
    ]

    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"{path.relative_to(PACKAGE_ROOT)} references {needle}"
