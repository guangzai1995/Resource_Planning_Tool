import sys

import pytest

import run_bench_multi as m


def test_parse_args_seed_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_bench_multi.py", "--model", "m"])

    args = m._parse_args()

    assert args.seed == 0
    assert args.no_vary_seed_by_config is False


def test_parse_args_seed_and_compatibility_flag(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_bench_multi.py",
            "--model",
            "m",
            "--seed",
            "456",
            "--no-vary-seed-by-config",
        ],
    )

    args = m._parse_args()

    assert args.seed == 456
    assert args.no_vary_seed_by_config is True


def test_derive_config_seed_is_stable():
    first = m.derive_config_seed(
        base_seed=0,
        input_len=4096,
        output_len=1024,
        parallel_num=8,
        prefix_ratio=0.8,
        config_index=3,
    )
    second = m.derive_config_seed(
        base_seed=0,
        input_len=4096,
        output_len=1024,
        parallel_num=8,
        prefix_ratio=0.8,
        config_index=3,
    )

    assert first == second
    assert 0 <= first < 2**32


def test_derive_config_seed_matches_stable_sha256_sample():
    # key: "0:4096:1024:8:0.8:3" -> sha256[:8] == "0685a547"
    assert m.derive_config_seed(
        base_seed=0,
        input_len=4096,
        output_len=1024,
        parallel_num=8,
        prefix_ratio=0.8,
        config_index=3,
    ) == 109421895


def test_derive_config_seed_changes_with_parallel():
    low_parallel = m.derive_config_seed(
        base_seed=0,
        input_len=4096,
        output_len=1024,
        parallel_num=1,
        prefix_ratio=0.8,
        config_index=3,
    )
    high_parallel = m.derive_config_seed(
        base_seed=0,
        input_len=4096,
        output_len=1024,
        parallel_num=8,
        prefix_ratio=0.8,
        config_index=3,
    )

    assert low_parallel != high_parallel


def test_derive_config_seed_changes_with_config_index():
    first_config = m.derive_config_seed(
        base_seed=0,
        input_len=4096,
        output_len=1024,
        parallel_num=8,
        prefix_ratio=0.8,
        config_index=1,
    )
    second_config = m.derive_config_seed(
        base_seed=0,
        input_len=4096,
        output_len=1024,
        parallel_num=8,
        prefix_ratio=0.8,
        config_index=2,
    )

    assert first_config != second_config


def test_derive_config_seed_changes_with_base_seed():
    first_base = m.derive_config_seed(
        base_seed=123,
        input_len=4096,
        output_len=1024,
        parallel_num=8,
        prefix_ratio=0.8,
        config_index=3,
    )
    second_base = m.derive_config_seed(
        base_seed=456,
        input_len=4096,
        output_len=1024,
        parallel_num=8,
        prefix_ratio=0.8,
        config_index=3,
    )

    assert first_base != second_base


@pytest.mark.parametrize(
    ("field", "override"),
    [
        ("input_len", 8192),
        ("output_len", 2048),
        ("prefix_ratio", 0.5),
        ("parallel_num", 16),
        ("config_index", 4),
        ("base_seed", 999),
    ],
)
def test_derive_config_seed_changes_when_any_key_field_changes(field, override):
    base = {
        "base_seed": 0,
        "input_len": 4096,
        "output_len": 1024,
        "parallel_num": 8,
        "prefix_ratio": 0.8,
        "config_index": 3,
    }
    changed = dict(base)
    changed[field] = override

    assert m.derive_config_seed(**base) != m.derive_config_seed(**changed)


def test_effective_config_seed_uses_base_seed_when_vary_disabled():
    assert m.effective_config_seed(
        base_seed=123,
        input_len=4096,
        output_len=1024,
        parallel_num=8,
        prefix_ratio=0.8,
        config_index=3,
        vary_seed_by_config=False,
    ) == 123


def test_effective_config_seed_delegates_to_derived_seed_when_vary_enabled():
    expected = m.derive_config_seed(
        base_seed=123,
        input_len=4096,
        output_len=1024,
        parallel_num=8,
        prefix_ratio=0.8,
        config_index=3,
    )

    assert m.effective_config_seed(
        base_seed=123,
        input_len=4096,
        output_len=1024,
        parallel_num=8,
        prefix_ratio=0.8,
        config_index=3,
        vary_seed_by_config=True,
    ) == expected


def test_validate_seed_accepts_valid_boundary_values():
    m.validate_seed(0)
    m.validate_seed(2**32 - 1)


def test_validate_seed_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="--seed"):
        m.validate_seed(-1)

    with pytest.raises(ValueError, match="--seed"):
        m.validate_seed(2**32)
