"""
base.py — the contract every synthesis backend implements
=========================================================
The pipeline in speak.py needs only one rendering method. Backends may own
processes, GPU weights, HTTP clients or anything else; none of that leaks into
the queue, playback or MCP tool surface.

The return value of synthesize() is deliberately small:

    wav          float PCM samples, 1-D or (1, N)
    sample_rate  integer Hz
    device       short device label for status/history
    elapsed_s    measured render time

Extra keys are allowed and are ignored by the pipeline. That gives a backend
room to report its own metadata without widening the core contract.
"""

from __future__ import annotations

from typing import Any, Protocol


class Backend(Protocol):
    """Minimum surface the worker expects from a synthesis backend."""

    def synthesize(self, text: str, model_key: str, **knobs: Any) -> dict[str, Any]:
        """Render one chunk and return its waveform plus metadata."""
        ...

    def status(self) -> dict[str, Any]:
        """Report readiness without loading weights or starting servers."""
        ...

    def unload(self) -> dict[str, Any]:
        """Release any resident weights or backend process."""
        ...

    def list_voices(self) -> dict[str, Any]:
        """Return this backend's available voices in a JSON-safe shape."""
        ...


__all__ = ["Backend"]
