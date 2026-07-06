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
        "references/config-guide.md",
        "reports/.gitkeep",
    ]

    for relative_path in expected_files:
        assert (PACKAGE_ROOT / relative_path).is_file(), relative_path

    assert not (PACKAGE_ROOT / "skills").exists()


def test_skill_frontmatter_is_valid_and_actionable():
    content = (PACKAGE_ROOT / "SKILL.md").read_text(encoding="utf-8")

    assert content.startswith("---\n")
    assert "name: generic-performance-testing" in content
    assert "description: 需要创建" in content
    assert "## 工作流程" in content
    assert "运行前确认需求" in content
    assert "references/config-guide.md" in content
    assert "scripts/run_auto.sh" in content


def test_config_guide_documents_required_setup_and_confirmation_flow():
    guide = (PACKAGE_ROOT / "references/config-guide.md").read_text(encoding="utf-8")

    required_sections = [
        "# 配置文件辅助指南",
        "## 运行前需求确认清单",
        "## 配置文件选择",
        "## 顶层字段说明",
        "## target.type 配置差异",
        "## 推荐运行流程",
    ]
    for section in required_sections:
        assert section in guide

    required_terms = [
        "测试目标",
        "接口类型",
        "base_url",
        "model",
        "认证",
        "数据集",
        "requests",
        "concurrency",
        "dry-run",
        "用户确认后",
    ]
    for term in required_terms:
        assert term in guide


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
