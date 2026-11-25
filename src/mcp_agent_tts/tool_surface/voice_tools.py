"""
voice_tools — the reference-clip inventory and playback control
==============================================================
list_voices:  what reference clips exist, so the model can pick one by name
              instead of inventing a path.
stop_speech:  cut off playback of a long utterance — the audible equivalent of
              interrupting the agent mid-sentence.
"""

from __future__ import annotations

import json

from mcp_agent_tts.backends.lfm3.registry import VOICES as LFM3_VOICES
from mcp_agent_tts.backends.registry import (
    BACKEND_CHATTERBOX,
    BACKEND_LFM3,
    BACKEND_SPECS,
    normalize_backend,
)
from mcp_agent_tts.errors import tool_error
from mcp_agent_tts.registry import VOICE_SUFFIXES, describe_models

# Aliased: the MCP tool below is itself called list_voices, and an unaliased
# import would be shadowed by it.
from mcp_agent_tts.voices import list_voices as _scan_voices
from mcp_agent_tts.worker import WorkerUnavailable, send_request


def register(server, ctx) -> None:
    voices_dir = ctx.cfg.voices_dir
    cfg = ctx.cfg

    def _worker(op: str) -> tuple[dict | None, str | None]:
        """Ask the worker. Returns (response, problem-description)."""
        try:
            return send_request({"op": op}, port=cfg.worker_port, timeout=cfg.worker_connect_timeout_s), None
        except WorkerUnavailable as exc:
            return None, str(exc)

    @server.tool()
    def list_voices(backend: str | None = None) -> str:
        """
        List the voices available for the selected backend.

        Chatterbox: the reference clips in the voices directory, each with the
        name you pass as voice=, its filename, size, duration, sample rate and
        channel count, plus which models can use it. The multilingual and
        original models require a clip; turbo/nano do not.

        LFM3: the two built-in voices, "female" and "male". LFM3 has no
        reference-audio cloning, so its voices are fixed rather than files on
        disk.

        For good Chatterbox cloning use as much clean single-speaker audio as
        you have, not a short excerpt — clipping a good recording down
        measurably adds fine clicks and crackle. A mastered or compressed source
        is fine; do not normalise it. Sample rate and channel count need not
        match anything. Clips too short to be safe carry a quality_note.
        """
        try:
            selected = normalize_backend(backend) or cfg.backend
            if selected not in BACKEND_SPECS:
                return json.dumps(
                    {
                        "status": "failed",
                        "reason": "unknown_backend",
                        "supported": sorted(BACKEND_SPECS),
                    },
                    indent=2,
                )
            if selected == BACKEND_LFM3:
                return json.dumps(
                    {
                        "status": "ok",
                        "backend": BACKEND_LFM3,
                        "count": len(LFM3_VOICES),
                        "voices": [
                            {
                                "name": name,
                                "fixed": True,
                                "requires_reference_clip": False,
                            }
                            for name in LFM3_VOICES
                        ],
                        "voice_cloning": False,
                        "note": "Pass female or male as voice= to speak().",
                    },
                    indent=2,
                    ensure_ascii=False,
                )

            voices = _scan_voices(voices_dir)
            return json.dumps(
                {
                    "status": "ok",
                    "backend": BACKEND_CHATTERBOX,
                    "voices_dir": str(voices_dir),
                    "count": len(voices),
                    "voices": voices,
                    "accepted_formats": list(VOICE_SUFFIXES),
                    "requires_reference_clip": [
                        m["key"]
                        for m in describe_models(BACKEND_CHATTERBOX)
                        if not m["stock_voice"]
                    ],
                    "note": (
                        "No reference clips yet — add a wav/mp3 to the voices "
                        "directory, or speak with model='nano' which ships a "
                        "built-in voice."
                        if not voices
                        else "Pass the name as voice= to speak()."
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        except Exception as e:
            return tool_error("list_voices", e)

    @server.tool()
    def stop_speech() -> str:
        """
        Stop the current speech and clear everything still queued. Cuts the
        sentence that is playing and cancels the remaining queued utterances,
        so a long reply does not resume a moment after it was interrupted.
        """
        try:
            response, problem = _worker("stop")
            if response is None:
                return json.dumps(
                    {
                        "status": "failed",
                        "reason": "worker_not_running",
                        "message": (
                            f"No synthesis worker is listening on port {cfg.worker_port} "
                            f"({problem}), so nothing is speaking."
                        ),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            return json.dumps({"status": "ok", **response}, indent=2, ensure_ascii=False)
        except Exception as e:
            return tool_error("stop_speech", e)


__all__ = ["register"]
