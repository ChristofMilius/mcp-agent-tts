"""
server.py — MCP server assembly
==============================
Wires Config into an AppContext, builds an MCPServer, and registers the tool
surface. Uses mcp 2.x (`mcp.server.mcpserver.MCPServer`).

The server owns no engine and no queue. Synthesis belongs to the resident
worker process (worker.py); this process carries text to it. That split exists
because background CUDA work inside the SDK's process stalled after two
sentences every time, over both transports, while the same code outside a
server finished the whole reply.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

from mcp.server.mcpserver import MCPServer

from mcp_agent_tts import __version__
from mcp_agent_tts.config import ENV_FILE, Config
from mcp_agent_tts.context import AppContext
from mcp_agent_tts.logging_setup import setup_logging
from mcp_agent_tts.tool_surface import register_all

logger = logging.getLogger(__name__)

SERVER_NAME = "mcp-agent-tts"

INSTRUCTIONS = """
Local text-to-speech through one MCP surface, with switchable backends running
on this machine. Text in, spoken audio out.

Picking a backend:
  * backend="chatterbox" (default) runs Resemble AI's nano/turbo/multilingual/
    original models. nano is the preferred default: 110M, English, built-in
    voice, inline tags ([laugh], [chuckle], [sigh], [gasp], [whispering],
    [clear throat]) and the tightest latency/memory budget. multilingual
    speaks 23 languages including German but REQUIRES a reference clip.
    original is the English CFG/exaggeration-tuning model and needs a clip.
  * backend="lfm3" runs Liquid AI's LFM3-Omni-2B GGUF through a resident
    llama-server /tts endpoint. It has fixed female/male voices, 24 kHz WAV
    output and no reference-audio cloning. Pass model="lfm3" to select it
    without naming a backend.

Voice rules:
  * Chatterbox: voice= / reference_clip= clone a voice on any model; nano/turbo
    fall back to their built-in voice when neither is given. multilingual and
    original refuse to render without a clip.
  * LFM3: voice must be female or male. A reference clip is rejected with
    voice_cloning_unsupported.

Workflow:
  1. Make sure the resident worker is running (`mcp-agent-tts worker`);
     speak() reports worker_not_running when it is not.
  2. tts_status() -> confirm backends, defaults, device, free VRAM per GPU and
     what the worker is doing. Loads nothing.
  3. list_voices() -> Chatterbox reference clips or LFM3's fixed voices.
  4. speak(text, backend=..., model=..., voice=..., language=...) -> hands the
     text to the worker, which renders chunk by chunk (up to
     MCP_TTS_SENTENCES_PER_CHUNK sentences each) and plays each one as soon as
     it is written. The call returns as soon as the text is accepted; playback
     continues in the resident worker even if the harness kills this tool call
     (LM Studio's tool-call timeout cannot cut speech off).
  5. tts_unload() -> evict Chatterbox weights and stop the LFM3 server. Safe
     when nothing is loaded.

Notes:
  * The first Chatterbox call downloads weights from Hugging Face; the first
    LFM3 call starts llama-server and loads its GGUF and mmproj. Both stay
    resident for the life of the worker.
  * Chatterbox style controls (exaggeration, cfg_weight, temperature) are
    optional; leave them unset so each model keeps its own tuned defaults.
  * On a machine where a large language model already occupies a GPU, check
    free VRAM with tts_status() before a long run of speak() calls.
""".strip()


def build_context() -> AppContext:
    """Construct the application object graph."""
    cfg = Config()
    setup_logging(str(cfg.logs_dir))
    return AppContext(cfg=cfg)


def create_server(ctx: AppContext | None = None) -> MCPServer:
    """Build an MCPServer with the full tool surface registered."""
    if ctx is None:
        ctx = build_context()

    server = MCPServer(
        name=SERVER_NAME,
        version=__version__,
        instructions=INSTRUCTIONS,
    )
    register_all(server, ctx)

    logger.info(
        "[server] %s v%s ready (pid=%d, ppid=%d, output=%s, voices=%s, "
        "backend=%s, model=%s, voice=%s, config=%s)",
        SERVER_NAME,
        __version__,
        os.getpid(),
        os.getppid(),
        ctx.cfg.output_dir,
        ctx.cfg.voices_dir,
        ctx.cfg.backend,
        ctx.cfg.model,
        ctx.cfg.voice or "unset",
        ENV_FILE if ENV_FILE.is_file() else "no .env - defaults in use",
    )
    return server


#: The transports MCPServer.run() accepts, matching the SDK's own Literal
#: exactly. Declaring it here (instead of passing a bare str) means the type
#: checker validates the value handed to the SDK -- no cast required.
Transport = Literal["stdio", "sse", "streamable-http"]
TRANSPORTS: tuple[Transport, ...] = ("stdio", "sse", "streamable-http")


def run(transport: Transport = "stdio", host: str = "127.0.0.1", port: int = 8000) -> None:
    """
    Build and run the server.

    transport: "stdio" (default, what opencode uses), "sse" or
               "streamable-http". host/port apply to the network transports.

    Nothing is drained on the way out: this process no longer owns synthesis.
    A reply that is still being spoken when the harness ends the turn keeps
    playing, because the resident worker is a different process.
    """
    # Reachable only from an untyped caller (argparse hands us a str); the
    # membership test narrows to Transport for the calls below.
    if transport not in TRANSPORTS:
        raise SystemExit(f"unknown transport {transport!r}; choose one of {', '.join(TRANSPORTS)}")
    ctx = build_context()
    server = create_server(ctx)
    try:
        if transport == "stdio":
            server.run(transport="stdio")
        else:
            server.run(transport=transport, host=host, port=port)
    except KeyboardInterrupt:
        pass


__all__ = [
    "SERVER_NAME",
    "TRANSPORTS",
    "Transport",
    "build_context",
    "create_server",
    "run",
]
