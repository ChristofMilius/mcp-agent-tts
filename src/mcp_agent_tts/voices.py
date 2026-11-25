"""
voices.py — compatibility façade for reference-clip discovery
=============================================================
Chatterbox's clip scanner moved into `backends.chatterbox.voices`. The generic
harness calls the backend directly; this module keeps the old import path
working for tests and for callers that only need clip lookup.
"""

from __future__ import annotations

from mcp_agent_tts.backends.chatterbox.voices import (
    SHORT_CLIP_S,
    list_voices,
    resolve_voice,
)

__all__ = ["SHORT_CLIP_S", "list_voices", "resolve_voice"]
