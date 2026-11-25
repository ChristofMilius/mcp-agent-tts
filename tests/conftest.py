"""Shared fixtures. Every test runs on the CPU with no model download."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from mcp_agent_tts.config import ENV_FILE_KEYS, Config
from mcp_agent_tts.speak import stop_all_speakers

# Env vars that Config reads. Cleared so a developer's own shell cannot leak
# into a test run and change the expected result.
_CONFIG_ENV = [
    "MCP_TTS_OUTPUT_DIR",
    "MCP_TTS_VOICES_DIR",
    "MCP_TTS_LOGS_DIR",
    "MCP_TTS_BACKEND",
    "MCP_TTS_MODEL",
    "MCP_TTS_VOICE",
    "MCP_TTS_LFM3_MODEL",
    "MCP_TTS_LFM3_MMPROJ",
    "MCP_TTS_LFM3_EXECUTABLE",
    "MCP_TTS_LFM3_SERVER_PORT",
    "MCP_TTS_LFM3_VOICE",
    "MCP_TTS_LFM3_LANGUAGE",
    "MCP_TTS_LFM3_NGL",
    "MCP_TTS_LFM3_CTX",
    "MCP_TTS_LFM3_STARTUP_TIMEOUT",
    "MCP_TTS_LFM3_REQUEST_TIMEOUT",
    "MCP_TTS_DEVICE",
    "MCP_TTS_GPU_INDEX",
    "MCP_TTS_VRAM_MB",
    "MCP_TTS_STRICT_VRAM",
    "MCP_TTS_AUTOPLAY",
    "MCP_TTS_HF_HOME",
    "MCP_TTS_MAX_CHARS",
    "MCP_TTS_MAX_CHUNK_CHARS",
    "MCP_TTS_SENTENCES_PER_CHUNK",
    "MCP_TTS_CHUNK_PAUSE_MS",
    "MCP_TTS_DRAIN_TIMEOUT",
    "MCP_TTS_WORKER_PORT",
    "MCP_TTS_WORKER_CONNECT_TIMEOUT",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # _CONFIG_ENV covers what a developer's shell might export; ENV_FILE_KEYS
    # covers what config.load_env_file() seeded from the real .env at import
    # time. Without the second set, a machine with a local .env would run these
    # tests against those values and the expected results would drift.
    for name in (*_CONFIG_ENV, *ENV_FILE_KEYS):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _chatterbox_backend_stubs(monkeypatch):
    """
    Let the ported Chatterbox tests run without the optional `chatterbox` extra.

    The production engine deliberately refuses to load when torch/chatterbox are
    absent; these tests stub only that dependency gate and the torch handle.
    Tests that need particular CUDA behaviour install their own fake afterwards.
    """
    from mcp_agent_tts.backends.chatterbox import engine as chatterbox_engine

    monkeypatch.setattr(chatterbox_engine, "_has_wheel", lambda _name: True)
    monkeypatch.setattr(
        chatterbox_engine, "_load_torch", lambda: FakeTorch(available=False, count=0)
    )


@pytest.fixture(autouse=True)
def _release_speakers():
    """
    Stop every Speaker worker a test started.

    Speaker's worker thread is non-daemon on purpose: a daemon thread is
    destroyed with the process, which is what silently truncated real speech
    when a harness ended the turn. The cost is that an unstopped worker keeps
    the interpreter alive, so without this the suite passes and then hangs.
    """
    yield
    stop_all_speakers()


@pytest.fixture
def cfg(tmp_path: Path) -> Iterator[Config]:
    """A Config rooted entirely inside tmp_path, with autoplay off."""
    monkey = pytest.MonkeyPatch()
    monkey.setenv("MCP_TTS_OUTPUT_DIR", str(tmp_path / "out"))
    monkey.setenv("MCP_TTS_VOICES_DIR", str(tmp_path / "voices"))
    monkey.setenv("MCP_TTS_LOGS_DIR", str(tmp_path / "logs"))
    monkey.setenv("MCP_TTS_AUTOPLAY", "0")
    try:
        yield Config()
    finally:
        monkey.undo()


class FakeTorch:
    """Just enough torch.cuda to drive device selection in the engine."""

    __version__ = "2.6.0+cu124"

    class version:  # noqa: N801 — mirrors torch.version.cuda
        cuda = "12.4"

    def __init__(self, *, available=True, count=1, free_mb=None, names=None):
        self.cuda = _FakeCuda(available, count, free_mb, names or {})
        self.manual_seed_calls: list[int] = []

    def manual_seed(self, seed: int) -> None:
        self.manual_seed_calls.append(seed)

    def __repr__(self):
        return f"<FakeTorch cuda={self.cuda}>"


class _FakeCuda:
    def __init__(self, available, count, free_mb, names):
        self._available = available
        self._count = count
        self._free = free_mb or [0] * count
        self._names = names
        self.manual_seed_all_calls: list[int] = []

    def is_available(self):
        return self._available

    def device_count(self):
        return self._count

    def mem_get_info(self, index):
        free = self._free[index] * 1024 * 1024
        return free, 12 * 1024 * 1024 * 1024

    def get_device_name(self, index):
        if self._names:
            return self._names.get(index, f"fake-gpu-{index}")
        return f"fake-gpu-{index}"

    def empty_cache(self):
        pass

    def manual_seed_all(self, seed: int) -> None:
        self.manual_seed_all_calls.append(seed)

    def __repr__(self):
        return f"cuda(available={self._available}, count={self._count})"


@pytest.fixture
def fake_torch(monkeypatch):
    """Install a FakeTorch in place of the real torch, and hand back a setter."""

    def _install(**kwargs):
        fake = FakeTorch(**kwargs)
        monkeypatch.setattr(
            "mcp_agent_tts.backends.chatterbox.engine._load_torch", lambda: fake
        )
        return fake

    return _install


class FakeEngine:
    """Stands in for ChatterboxEngine. Records calls, returns a tiny waveform."""

    def __init__(self, sample_rate=24000, samples=2400):
        self.calls = []
        self.wav = np.zeros((1, samples), dtype=np.float32)
        self.sample_rate = sample_rate

    def synthesize(self, text, model_key, **kwargs):
        self.calls.append({"text": text, "model": model_key, **kwargs})
        return {
            "wav": self.wav,
            "sample_rate": self.sample_rate,
            "model": model_key,
            "device": "cpu",
            "chars": len(text),
            "elapsed_s": 0.01,
            "language": kwargs.get("language"),
            "reference_clip_used": bool(kwargs.get("reference_clip")),
            "free_vram_mb": None,
        }


@pytest.fixture
def fake_engine():
    return FakeEngine()


@pytest.fixture
def silent_playback(monkeypatch):
    """Never touch the speakers during a test run."""
    monkeypatch.setattr("mcp_agent_tts.speak.play", lambda *a, **k: (False, "stubbed"))
    return True


def pytest_report_header(config):
    return f"python: {sys.version.split()[0]}"
