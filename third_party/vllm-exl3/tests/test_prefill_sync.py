"""CPU tests for EXL3 prefill synchronization workaround."""

import pytest

torch = pytest.importorskip("torch")

import vllm_exl3.exl3 as exl3


class TestEnvParsing:
    """Test that the env var VLLM_EXL3_PREFILL_SYNC is parsed correctly."""

    def test_env_parsing_valid_integer(self, monkeypatch):
        """Parse valid integer from env var."""
        monkeypatch.setenv("VLLM_EXL3_PREFILL_SYNC", "256")
        assert exl3._env_prefill_sync_rows() == 256

    def test_env_parsing_invalid_string(self, monkeypatch):
        """Non-numeric string returns 256 (default fallback)."""
        monkeypatch.setenv("VLLM_EXL3_PREFILL_SYNC", "abc")
        assert exl3._env_prefill_sync_rows() == 256

    def test_env_parsing_negative(self, monkeypatch):
        """Negative value clamps to 0."""
        monkeypatch.setenv("VLLM_EXL3_PREFILL_SYNC", "-5")
        assert exl3._env_prefill_sync_rows() == 0

    def test_env_parsing_unset(self, monkeypatch):
        """Unset env var returns 0."""
        monkeypatch.delenv("VLLM_EXL3_PREFILL_SYNC", raising=False)
        assert exl3._env_prefill_sync_rows() == 0


class TestSyncLogic:
    """Test that _prefill_sync calls synchronize only for rows in the expected range."""

    def test_sync_only_for_prefill_rows_in_range(self, monkeypatch):
        """Sync only for row counts in 2..max_rows range."""
        monkeypatch.setattr(exl3, "_EXL3_PREFILL_SYNC", 256)

        # Record all synchronize calls.
        calls = []
        def record_sync():
            calls.append(True)
        monkeypatch.setattr(exl3.torch.cuda, "synchronize", record_sync)
        monkeypatch.setattr(exl3.torch.cuda, "is_current_stream_capturing", lambda: False)

        # Single row (decode): no sync.
        exl3._prefill_sync(1)
        assert len(calls) == 0

        # In range (72 is between 2 and 256): sync.
        exl3._prefill_sync(72)
        assert len(calls) == 1

        # Out of range (257 > 256): no sync.
        exl3._prefill_sync(257)
        assert len(calls) == 1

    def test_no_sync_during_graph_capture(self, monkeypatch):
        """Never sync if inside CUDA graph capture."""
        monkeypatch.setattr(exl3, "_EXL3_PREFILL_SYNC", 256)

        calls = []
        def record_sync():
            calls.append(True)
        monkeypatch.setattr(exl3.torch.cuda, "synchronize", record_sync)
        monkeypatch.setattr(exl3.torch.cuda, "is_current_stream_capturing", lambda: True)

        # Even though 72 is in range, no sync during graph capture.
        exl3._prefill_sync(72)
        assert len(calls) == 0

    def test_disabled_by_default(self, monkeypatch):
        """When _EXL3_PREFILL_SYNC is 0, no sync even for in-range rows."""
        monkeypatch.setattr(exl3, "_EXL3_PREFILL_SYNC", 0)

        calls = []
        def record_sync():
            calls.append(True)
        monkeypatch.setattr(exl3.torch.cuda, "synchronize", record_sync)
        monkeypatch.setattr(exl3.torch.cuda, "is_current_stream_capturing", lambda: False)

        # No sync when disabled.
        exl3._prefill_sync(72)
        assert len(calls) == 0
