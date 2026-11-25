"""
mcp_agent_tts — MCP server for local text-to-speech
==================================================
One agent-facing tool surface, several switchable synthesis backends. The
queueing, sentence chunking, render-ahead playback, WAV output, history and MCP
tools live here once; each backend owns only its engine and model rules.

Backends in this release:

  * chatterbox (default) — Resemble AI's nano (110M), turbo (350M),
    multilingual (500M, 23 languages) and original (500M). nano/turbo have a
    stock voice and understand inline paralinguistic tags; multilingual and
    original require a reference clip for voice cloning.
  * lfm3 — Liquid AI's LFM3-Omni-2B GGUF through a resident llama-server
    `/tts` endpoint. Fixed female/male voices, 24 kHz WAV, no voice cloning.

Design notes worth knowing:

  * Synthesis runs in a separate, long-lived worker process, not in the MCP
    server (see worker.py). The server holds no engine and never loads
    weights; it forwards speak() to the worker, which boots in milliseconds,
    loads the selected backend on the first utterance and keeps it resident.

  * Backend instances are owned by TtsEngine (engine.py). The pipeline depends
    only on synthesize()/status()/unload()/list_voices(); backend-specific
    validation stays in the tool surface and backend-specific rendering stays
    in `backends/`.

  * Chatterbox keeps one resident model at a time (see MAX_RESIDENT). On a 12 GB
    card that is the difference between working and a CUDA OOM. Switching
    models evicts the previous one and empties the allocator cache.

  * The CUDA wheel index is pinned in pyproject.toml behind the `chatterbox`
    extra, so that backend's GPU acceleration is reproducible rather than a
    platform accident. LFM3 needs no Python packages; llama.cpp serves it.
"""

from __future__ import annotations

__version__ = "0.1.0"


def main() -> int:
    """Console entry point (`mcp-agent-tts`). Dispatches to the CLI."""
    from mcp_agent_tts.cli import main as _cli_main

    return _cli_main()


__all__ = ["__version__", "main"]
