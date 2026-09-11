"""Tests for the batch-adaptive EXL3 speculative draft scheduler."""

import pytest

from vllm_exl3.exl3 import (
    DEFAULT_SPECULATIVE_SCHEDULE,
    get_speculative_draft_tokens,
    parse_speculative_schedule,
)


@pytest.mark.parametrize(
    ("batch_size", "expected"),
    [
        (1, 3),
        (4, 3),
        (5, 2),
        (8, 2),
        (9, 1),
        (16, 1),
        (17, 0),
        (32, 0),
        (0, 0),
        (-1, 0),
    ],
)
def test_default_schedule_boundaries(batch_size, expected):
    assert get_speculative_draft_tokens(batch_size) == expected


def test_custom_schedule_argument_overrides_default():
    custom = [[1, 2, 5], (3, 6, 2)]
    assert get_speculative_draft_tokens(1, custom) == 5
    assert get_speculative_draft_tokens(2, custom) == 5
    assert get_speculative_draft_tokens(3, custom) == 2
    assert get_speculative_draft_tokens(6, custom) == 2
    assert get_speculative_draft_tokens(7, custom) == 0


def test_environment_schedule(monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_SPEC_SCHEDULE", "1:1:6,2:3:4")
    assert get_speculative_draft_tokens(1) == 6
    assert get_speculative_draft_tokens(2) == 4
    assert get_speculative_draft_tokens(4) == 0


@pytest.mark.parametrize(
    "malformed",
    [
        "",
        "not-a-schedule",
        "1:4",
        "1:4:three",
        "1:4:3,4:8:2",  # overlapping ranges
        "0:4:3",
        "5:1:2",
        "1:4:-1",
    ],
)
def test_malformed_schedule_falls_back_to_defaults(malformed):
    assert parse_speculative_schedule(malformed) == DEFAULT_SPECULATIVE_SCHEDULE


def test_malformed_environment_falls_back_to_defaults(monkeypatch):
    monkeypatch.setenv("VLLM_EXL3_SPEC_SCHEDULE", "1:4:bad")
    assert get_speculative_draft_tokens(1) == 3
    assert get_speculative_draft_tokens(8) == 2


def test_default_schedule_result_is_not_mutable_global():
    parsed = parse_speculative_schedule("")
    parsed.append((20, 20, 9))
    assert parse_speculative_schedule("") == DEFAULT_SPECULATIVE_SCHEDULE
