from pathlib import Path
import stat


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
        "tests/test_package_structure.py",
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


def test_automated_skill_prefers_package_script_usage():
    text = (
        ROOT / "skills/automated-performance-testing/SKILL.md"
    ).read_text(encoding="utf-8")
    assert "## Preferred script usage" in text
    assert (
        "python3 scripts/perf_auto.py --config configs/openai_chat.json --dry-run"
        in text
    )
    assert "ad-hoc" in text


def test_readme_documents_portable_package_usage():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    lowered = text.lower()

    assert "portable package" in lowered
    assert "independent" in lowered
    assert "does not depend on this repository" in lowered
    assert "copy" in lowered
    for directory in ["skills/", "configs/", "datasets/", "scripts/", "reports/", "tests/"]:
        assert directory in text
    for protocol in [
        "openai_chat",
        "openai_completion",
        "openai_asr",
        "generic_http",
    ]:
        assert protocol in text
    for command in [
        "./scripts/run_manual.sh --config configs/openai_chat.json --dry-run",
        "./scripts/run_manual.sh --config configs/openai_chat.json --mode curl",
        "./scripts/run_auto.sh --config configs/openai_chat.json --dry-run",
        "./scripts/run_auto.sh --config configs/openai_chat.json --concurrency 1,2 --epochs 2",
    ]:
        assert command in text
    for report_name in [
        "summary.md",
        "metrics.json",
        "metrics.csv",
        "requests.jsonl",
        "errors.jsonl",
    ]:
        assert report_name in text
    assert "100% request failure" in text


def test_shell_wrappers_are_executable():
    for script_name in ["scripts/run_manual.sh", "scripts/run_auto.sh"]:
        mode = (ROOT / script_name).stat().st_mode
        assert mode & stat.S_IXUSR, f"{script_name} is not executable by the owner"


def test_manual_skill_documents_current_cli_modes():
    text = (
        ROOT / "skills/manual-interface-performance-testing/SKILL.md"
    ).read_text(encoding="utf-8")

    assert "--mode smoke" not in text
    assert "--mode curl" in text
    assert "--dry-run" in text


def test_automated_skill_documents_run_auto_and_failure_safety():
    text = (
        ROOT / "skills/automated-performance-testing/SKILL.md"
    ).read_text(encoding="utf-8")
    lowered = text.lower()

    assert "run_auto.sh" in text
    assert "fail-fast" in text
    assert "100% request failure" in lowered


def test_docs_clarify_p90_threshold_does_not_fail_fast():
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    automated_skill = (
        ROOT / "skills/automated-performance-testing/SKILL.md"
    ).read_text(encoding="utf-8").lower()

    for text in [readme, automated_skill]:
        assert "p90 threshold does not trigger fail-fast" in text
