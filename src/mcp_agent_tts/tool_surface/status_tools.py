"""
status_tools — readiness and VRAM reclamation
=============================================
tts_status: which models exist, which device would be used, how much VRAM is
            free on each card, whether weights are already resident, and what
            the resident worker is doing. Loads nothing, so it is safe to call
            as a preflight.
tts_unload:  drop the resident model and give the VRAM back. The escape hatch
            that matters on a shared GPU.

Both ask the worker, because the worker is what holds the model. They still
answer for a worker that is not running: the model and GPU inventory comes from
the registry and the local configuration, which is exactly the information you
want when the worker is the thing that is missing.
"""

from __future__ import annotations

import json

from mcp_agent_tts.errors import tool_error
from mcp_agent_tts.registry import (
    SUPPORTED_LANGUAGES,
    T3_MODELS,
    TURBO_TAGS,
    describe_backends,
    describe_models,
)
from mcp_agent_tts.worker import WorkerUnavailable, send_request


def register(server, ctx) -> None:
    cfg = ctx.cfg

    def _worker(op: str) -> tuple[dict | None, str | None]:
        """Ask the worker. Returns (response, problem-description)."""
        try:
            return send_request({"op": op}, port=cfg.worker_port, timeout=cfg.worker_connect_timeout_s), None
        except WorkerUnavailable as exc:
            return None, str(exc)

    @server.tool()
    def tts_status() -> str:
        """
        Report local TTS readiness: which backends are available, whether the
        resident synthesis worker is running, the models and languages each
        backend speaks, the configured defaults, the paralinguistic tags
        Chatterbox turbo understands, and — when the worker is up — the resolved
        device, per-GPU free VRAM and the speech queue (how many utterances are
        queued or rendering, plus the result of the last one).

        Loads no weights and starts nothing. On a machine where a large model
        already occupies a GPU, check free_mb first: a card with less than
        ~4096 MB free will fail to load.

        If worker_running is false, the speech tools will not work: tell the
        user to start `mcp-agent-tts worker`. Everything else in the
        response is still accurate.
        """
        try:
            response, problem = _worker("status")
            payload = {
                "status": "ok",
                "worker_running": response is not None,
                "backends": describe_backends(),
                "models": describe_models(),
                "default_backend": cfg.backend,
                "default_voice": cfg.voice or None,
                "default_model": cfg.model,
                "lfm3_configured": bool(cfg.lfm3_model and cfg.lfm3_mmproj),
                "lfm3_voice": cfg.lfm3_voice,
                "lfm3_language": cfg.lfm3_language,
                "worker_port": cfg.worker_port,
                "multilingual_languages": SUPPORTED_LANGUAGES,
                "multilingual_t3_models": sorted(T3_MODELS),
                "turbo_paralinguistic_tags": list(TURBO_TAGS),
            }
            if response is None:
                payload["worker_note"] = (
                    f"No synthesis worker is listening on port {cfg.worker_port} "
                    f"({problem}). Start one with `mcp-agent-tts worker`; "
                    "until then speak() reports worker_not_running."
                )
            else:
                runtime = dict(response)
                runtime.pop("ok", None)
                payload["worker"] = runtime
            return json.dumps(payload, indent=2, ensure_ascii=False)
        except Exception as e:
            return tool_error("tts_status", e)

    @server.tool()
    def tts_unload() -> str:
        """
        Release every instantiated backend from the resident worker: evict
        Chatterbox weights from VRAM and stop a running LFM3 llama-server. Use
        this when another process needs the card, or before loading a different
        model. Safe to call when nothing is loaded.
        """
        try:
            response, problem = _worker("unload")
            if response is None:
                return json.dumps(
                    {
                        "status": "failed",
                        "reason": "worker_not_running",
                        "message": (
                            f"No synthesis worker is listening on port {cfg.worker_port} "
                            f"({problem}), so there is no model to release. A worker "
                            "frees its model on the way out unless it is still speaking."
                        ),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            return json.dumps({"status": "ok", **response}, indent=2, ensure_ascii=False)
        except Exception as e:
            return tool_error("tts_unload", e)


__all__ = ["register"]
