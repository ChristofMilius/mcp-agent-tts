"""
engine.py — backend dispatcher for the generic speech pipeline
==============================================================
The pipeline holds one object with a `synthesize()` method. That object used to
be ChatterboxEngine; now it is this thin router. It owns backend instances
lazily, picks one per request, and exposes the few status properties the queue
already expects.

No synthesis logic lives here. Backend-specific work stays in
`backends/chatterbox` and `backends/lfm3`; backend-neutral work stays in the
pipeline modules.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp_agent_tts.backends.registry import (
    BACKEND_CHATTERBOX,
    BACKEND_SPECS,
    describe_backends,
    describe_models,
    load_backend,
    resolve_backend,
)
from mcp_agent_tts.config import Config

logger = logging.getLogger(__name__)


class TtsEngine:
    """Route one pipeline `synthesize()` call to the selected backend."""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self._engines: dict[str, Any] = {}
        self._last_backend: str = self.cfg.backend if self.cfg.backend in BACKEND_SPECS else "chatterbox"

    # -- backend ownership ----------------------------------------------------
    def _engine_for(self, backend: str):
        if backend not in BACKEND_SPECS:
            backend = "chatterbox"
        engine = self._engines.get(backend)
        if engine is None:
            spec = BACKEND_SPECS[backend]
            engine = load_backend(spec)(self.cfg)
            self._engines[backend] = engine
            logger.info("[engine] %s backend instantiated", backend)
        return engine

    def backend_name(self, backend: str | None = None, model_key: str | None = None) -> str:
        """Resolve a request to a backend key without instantiating it."""
        return resolve_backend(backend, model_key, self.cfg.backend)

    # -- Backend contract -----------------------------------------------------
    def synthesize(self, text: str, model_key: str, **knobs: Any) -> dict[str, Any]:
        requested = knobs.pop("backend", None)
        backend = self.backend_name(requested, model_key)
        self._last_backend = backend
        engine = self._engine_for(backend)
        result = dict(engine.synthesize(text, model_key, **knobs))
        result.setdefault("backend", backend)
        result.setdefault("model", model_key)
        return result

    def status(self) -> dict[str, Any]:
        """Report both backends without loading weights or starting a server."""
        engines: dict[str, Any] = {}
        for name in BACKEND_SPECS:
            try:
                engines[name] = self._engine_for(name).status()
            except Exception as exc:  # noqa: BLE001 — status must never throw
                engines[name] = {"backend": name, "status": "error", "error": type(exc).__name__}

        active = engines.get(self._last_backend, {})
        return {
            "default_backend": self.cfg.backend,
            "active_backend": self._last_backend,
            "backends": describe_backends(),
            "models": describe_models(),
            "engines": engines,
            # Flatten the queue's usual fields from the active backend so the
            # worker and CLI keep working unchanged.
            "model_loaded": bool(active.get("loaded_model")),
            "loaded_model": active.get("loaded_model"),
            "model_loading": active.get("model_loading"),
            "device": active.get("device"),
            "free_vram_mb": active.get("free_vram_mb"),
            "vram_floor_mb": active.get("vram_floor_mb"),
            "strict_vram": active.get("strict_vram"),
            "autoplay": self.cfg.autoplay,
            "output_dir": str(self.cfg.output_dir),
            "voices_dir": str(self.cfg.voices_dir),
            "default_model": self.cfg.model,
            "max_chars": self.cfg.max_chars,
        }

    def unload(self) -> dict[str, Any]:
        """Release every backend that was instantiated, not just the active one."""
        results: dict[str, Any] = {}
        for name, engine in self._engines.items():
            unload = getattr(engine, "unload", None)
            if unload is None:
                results[name] = {"unloaded": None, "unsupported": True}
                continue
            try:
                results[name] = unload()
            except Exception as exc:  # noqa: BLE001 — one backend must not block the other
                logger.error("[engine] %s unload failed: %s", name, exc, exc_info=True)
                results[name] = {"unloaded": None, "error": type(exc).__name__}
        self._engines.clear()
        return {"unloaded_backends": results, "freed": True}

    def list_voices(self, backend: str | None = None) -> dict[str, Any]:
        name = backend or self._last_backend
        return self._engine_for(name).list_voices()

    def gpu_report(self) -> list[dict[str, Any]]:
        """Forward the CUDA inventory for `doctor`; LFM3 has no torch device."""
        report = getattr(self._engine_for(BACKEND_CHATTERBOX), "gpu_report", None)
        return list(report()) if callable(report) else []

    # -- properties used by Speaker.status() ----------------------------------
    def is_loaded(self) -> bool:
        engine = self._engines.get(self._last_backend)
        if engine is None:
            return False
        probe = getattr(engine, "is_loaded", None)
        if callable(probe):
            return bool(probe())
        return bool(getattr(engine, "status", lambda: {})().get("loaded_model"))

    @property
    def loading_model(self) -> str | None:
        engine = self._engines.get(self._last_backend)
        return getattr(engine, "loading_model", None) if engine is not None else None

    @property
    def loaded_model(self) -> str | None:
        engine = self._engines.get(self._last_backend)
        if engine is None:
            return None
        value = getattr(engine, "loaded_model", None)
        if value is not None:
            return value
        return engine.status().get("loaded_model")


__all__ = ["TtsEngine"]
