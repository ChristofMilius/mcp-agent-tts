"""
lfm3 engine — LFM3-Omni-2B through a resident llama-server
==========================================================
LFM3's GGUF build is served by llama.cpp. The engine keeps one `llama-server`
process alive and sends each rendered chunk to its `/tts` endpoint, so weights
are loaded once and every chunk comes back as 24 kHz WAV bytes.

Why a resident server instead of `llama-tts` per chunk: the CLI reloads the
model on every call. The pipeline renders several chunks ahead while the
previous chunk plays, so that reload cost would show up as gaps in speech.

The backend owns no torch objects. It returns a plain numpy PCM array, which
the generic audio module writes as WAV. No voice cloning is attempted: LFM3
exposes fixed `female` and `male` voices only.
"""

from __future__ import annotations

import io
import json
import logging
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

import numpy as np

from mcp_agent_tts.backends.lfm3.registry import SPEECH_LANGUAGES, VOICES, get_spec, normalize_voice
from mcp_agent_tts.config import Config
from mcp_agent_tts.errors import ToolFault

logger = logging.getLogger(__name__)


class Lfm3Engine:
    """Own one llama-server process and satisfy the pipeline's Backend contract."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self._proc: subprocess.Popen | None = None
        self._pid: int | None = None
        self._ready: bool = False
        self._base_url = f"http://127.0.0.1:{self.cfg.lfm3_server_port}"

    # -- helpers --------------------------------------------------------------
    def _voice(self, requested: str | None) -> str:
        value = normalize_voice(requested) or normalize_voice(self.cfg.lfm3_voice) or "male"
        if value not in VOICES:
            raise ToolFault(
                "unsupported_voice",
                f"LFM3 voices are {', '.join(VOICES)}; got {requested!r}.",
                supported=list(VOICES),
            )
        return value

    def _language(self, requested: str | None) -> str:
        value = (requested or self.cfg.lfm3_language or "en").strip().lower()
        if not value:
            value = "en"
        if value not in SPEECH_LANGUAGES:
            raise ToolFault(
                "unsupported_language",
                f"{requested!r} is not an LFM3 speech language. "
                f"Call tts_status for the supported codes.",
                supported=sorted(SPEECH_LANGUAGES),
            )
        return value

    def _paths(self) -> tuple[Path, Path]:
        raw_model = (self.cfg.lfm3_model or "").strip()
        raw_mmproj = (self.cfg.lfm3_mmproj or "").strip()
        if not raw_model or not raw_mmproj:
            raise ToolFault(
                "lfm3_not_configured",
                "LFM3 needs MCP_TTS_LFM3_MODEL and MCP_TTS_LFM3_MMPROJ in the project .env.",
                required=["MCP_TTS_LFM3_MODEL", "MCP_TTS_LFM3_MMPROJ"],
            )
        model = Path(raw_model).expanduser()
        mmproj = Path(raw_mmproj).expanduser()
        if not model.is_file() or not mmproj.is_file():
            raise ToolFault(
                "lfm3_files_missing",
                "The configured LFM3 model or mmproj file is missing. Check the .env paths.",
            )
        return model, mmproj

    def _executable(self) -> str:
        requested = (self.cfg.lfm3_executable or "llama-server").strip() or "llama-server"
        found = shutil.which(requested)
        if found is None:
            raise ToolFault(
                "lfm3_executable_missing",
                "No llama-server executable was found. Install llama.cpp or set "
                "MCP_TTS_LFM3_EXECUTABLE.",
            )
        return found

    def _props(self, timeout: float = 1.0) -> dict | None:
        """Read /props without starting anything. None means unavailable."""
        try:
            with urllib.request.urlopen(f"{self._base_url}/props", timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError, OSError):
            return None
        return payload if isinstance(payload, dict) else None

    def _ensure_server(self) -> None:
        if self._ready and self._proc is not None and self._proc.poll() is None:
            return
        if self._ready and self._proc is None and self._props() is not None:
            return

        props = self._props()
        if props is not None:
            voices = props.get("voices")
            if isinstance(voices, list) and set(voices) >= {"female", "male"}:
                logger.info("[lfm3] reusing resident server on port %d", self.cfg.lfm3_server_port)
                self._ready = True
                return
            raise ToolFault(
                "lfm3_port_conflict",
                f"Port {self.cfg.lfm3_server_port} answers but is not an LFM3 TTS server.",
            )

        model, mmproj = self._paths()
        executable = self._executable()
        voice = self._voice(None)
        language = self._language(None)

        args = [
            executable,
            "-m",
            str(model),
            "-mm",
            str(mmproj),
            "-ngl",
            str(self.cfg.lfm3_ngl),
            "-c",
            str(self.cfg.lfm3_ctx),
            "--tts-lang",
            language,
            "--tts-voice",
            voice,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.cfg.lfm3_server_port),
            "--no-ui",
        ]

        log_dir = Path(self.cfg.logs_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        out_path = log_dir / "lfm3-server.log"
        err_path = log_dir / "lfm3-server.err.log"
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        logger.info("[lfm3] starting llama-server on port %d", self.cfg.lfm3_server_port)
        with out_path.open("ab") as out, err_path.open("ab") as err:
            self._proc = subprocess.Popen(
                args,
                stdout=out,
                stderr=err,
                creationflags=creation_flags,
            )
        self._pid = self._proc.pid

        deadline = time.monotonic() + self.cfg.lfm3_startup_timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                self._proc = None
                self._pid = None
                raise ToolFault(
                    "lfm3_server_exited",
                    "llama-server exited during startup. Check the LFM3 server log.",
                )
            props = self._props(timeout=1.0)
            if props is not None:
                self._ready = True
                logger.info("[lfm3] server ready on port %d", self.cfg.lfm3_server_port)
                return
            time.sleep(0.25)

        self.unload()
        raise ToolFault(
            "lfm3_server_timeout",
            "llama-server did not become ready before the startup timeout. "
            "Check the LFM3 server log.",
        )

    def _post_tts(self, text: str, voice: str, language: str) -> bytes:
        body = json.dumps(
            {
                "input": text,
                "lang": language,
                "voice": voice,
                "response_format": "wav",
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/tts",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.cfg.lfm3_request_timeout_s
            ) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read(512).decode("utf-8", "replace") if exc.fp else ""
            logger.error("[lfm3] /tts returned HTTP %d: %s", exc.code, detail)
            raise ToolFault(
                "lfm3_synthesis_failed",
                f"LFM3 server returned HTTP {exc.code}. Check the LFM3 server log.",
                status_code=exc.code,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.error("[lfm3] /tts request failed: %s", exc)
            self._ready = False
            raise ToolFault(
                "lfm3_unavailable",
                "The LFM3 server became unreachable. The next call will restart it.",
            ) from exc

    @staticmethod
    def _decode_wav(payload: bytes) -> tuple[np.ndarray, int]:
        try:
            with wave.open(io.BytesIO(payload), "rb") as reader:
                sample_rate = reader.getframerate()
                width = reader.getsampwidth()
                channels = reader.getnchannels()
                frames = reader.readframes(reader.getnframes())
        except (wave.Error, EOFError) as exc:
            raise ToolFault(
                "lfm3_bad_audio",
                "LFM3 server returned a WAV response the harness could not read.",
            ) from exc
        if width != 2:
            raise ToolFault(
                "lfm3_bad_audio",
                f"Expected 16-bit PCM from LFM3, got {width * 8}-bit.",
            )
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        return np.ascontiguousarray(samples), sample_rate

    # -- Backend contract -----------------------------------------------------
    def synthesize(self, text: str, model_key: str, **knobs: Any) -> dict[str, Any]:
        spec = get_spec(model_key)
        if spec is None:
            raise ToolFault(
                "unknown_model",
                f"Unknown LFM3 model {model_key!r}. Use 'lfm3'.",
            )
        if knobs.get("reference_clip"):
            raise ToolFault(
                "voice_cloning_unsupported",
                "LFM3 has no reference-audio voice cloning. Use voice='female' or 'male'.",
            )

        voice = self._voice(knobs.get("voice"))
        language = self._language(knobs.get("language"))
        self._ensure_server()

        started = time.perf_counter()
        payload = self._post_tts(text, voice, language)
        wav, sample_rate = self._decode_wav(payload)
        elapsed = time.perf_counter() - started

        logger.info(
            "[lfm3] synthesized %d chars in %.2fs (%d samples @ %d Hz, voice=%s)",
            len(text),
            elapsed,
            len(wav),
            sample_rate,
            voice,
        )
        return {
            "wav": wav,
            "sample_rate": int(sample_rate),
            "model": spec.key,
            "backend": "lfm3",
            "device": f"llama-server:{self.cfg.lfm3_server_port}",
            "elapsed_s": round(elapsed, 2),
            "chars": len(text),
            "language": language,
            "voice": voice,
            "reference_clip_used": False,
        }

    def status(self) -> dict[str, Any]:
        props = self._props()
        running = self._proc is not None and self._proc.poll() is None
        return {
            "backend": "lfm3",
            "configured": bool(self.cfg.lfm3_model and self.cfg.lfm3_mmproj),
            "model_configured": bool(self.cfg.lfm3_model),
            "mmproj_configured": bool(self.cfg.lfm3_mmproj),
            "executable": self.cfg.lfm3_executable,
            "port": self.cfg.lfm3_server_port,
            "server_running": bool(running or props),
            "process_pid": self._pid if running else None,
            "voices": props.get("voices") if props else list(VOICES),
            "default_voice": self.cfg.lfm3_voice,
            "default_language": self.cfg.lfm3_language,
            "language_count": len(SPEECH_LANGUAGES),
            "sample_rate": 24000,
            "voice_cloning": False,
            "loaded_model": "lfm3" if props else None,
            "model_loading": None,
            "supports_reference_clip": False,
        }

    def unload(self) -> dict[str, Any]:
        was_running = self._proc is not None and self._proc.poll() is None
        pid = self._pid
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    self._proc.kill()
                except OSError:
                    pass
            self._proc = None
            self._pid = None
        self._ready = False
        return {
            "backend": "lfm3",
            "unloaded": "lfm3" if was_running else None,
            "was_loaded": was_running,
            "process_pid": pid,
            "port": self.cfg.lfm3_server_port,
        }

    def list_voices(self) -> dict[str, Any]:
        return {
            "backend": "lfm3",
            "voices": list(VOICES),
            "fixed": True,
            "voice_cloning": False,
            "note": "LFM3 speaks with one of two built-in voices; reference clips are ignored.",
        }


__all__ = ["Lfm3Engine"]
