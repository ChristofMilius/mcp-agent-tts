"""
backend registry — what can be switched, and what each backend can do
=====================================================================
The corpus pipeline does not know how a model renders audio. It asks this
module two questions: which backend handles a request, and what that backend's
model/voice/language rules are.

This is a static table rather than a plugin system. Two backends do not justify
discovery, entry points or a lifecycle API; a dict and a lazy import are enough.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from mcp_agent_tts.backends.chatterbox import registry as chatterbox_registry
from mcp_agent_tts.backends.lfm3 import registry as lfm3_registry

BACKEND_CHATTERBOX = "chatterbox"
BACKEND_LFM3 = "lfm3"

_BACKEND_ALIASES = {
    "cb": BACKEND_CHATTERBOX,
    "chatbox": BACKEND_CHATTERBOX,
    "liquid": BACKEND_LFM3,
    "lfm3-omni": BACKEND_LFM3,
    "lfm3-omni-2b": BACKEND_LFM3,
}


@dataclass(frozen=True)
class BackendSpec:
    """Discovery metadata for one synthesis backend."""

    key: str
    label: str
    engine_module: str
    engine_attr: str
    default_model: str
    model_specs: dict
    stock_voice: bool
    cloning: bool
    note: str


BACKEND_SPECS: dict[str, BackendSpec] = {
    BACKEND_CHATTERBOX: BackendSpec(
        key=BACKEND_CHATTERBOX,
        label="Chatterbox",
        engine_module="mcp_agent_tts.backends.chatterbox.engine",
        engine_attr="ChatterboxEngine",
        default_model="nano",
        model_specs=chatterbox_registry.MODEL_SPECS,
        stock_voice=True,
        cloning=True,
        note=(
            "Resemble AI Chatterbox family. Reference-clip voice cloning and "
            "inline paralinguistic tags on the models that support them. "
            "Install the `chatterbox` extra to enable it."
        ),
    ),
    BACKEND_LFM3: BackendSpec(
        key=BACKEND_LFM3,
        label="LFM3-Omni",
        engine_module="mcp_agent_tts.backends.lfm3.engine",
        engine_attr="Lfm3Engine",
        default_model="lfm3",
        model_specs=lfm3_registry.MODEL_SPECS,
        stock_voice=True,
        cloning=False,
        note=(
            "Liquid AI LFM3-Omni-2B GGUF through a resident llama-server /tts "
            "endpoint. Fixed female/male voices, 24 kHz WAV, no reference-audio "
            "cloning. Needs MCP_TTS_LFM3_MODEL and MCP_TTS_LFM3_MMPROJ."
        ),
    ),
}

#: Union of every backend's language table, for status listing only. Per-call
#: validation happens against the selected backend's own table.
SUPPORTED_LANGUAGES: dict[str, str] = {
    **chatterbox_registry.SUPPORTED_LANGUAGES,
    **lfm3_registry.SPEECH_LANGUAGES,
}

T3_MODELS = chatterbox_registry.T3_MODELS
VOICE_SUFFIXES = chatterbox_registry.VOICE_SUFFIXES


def normalize_backend(backend: str | None) -> str | None:
    """Normalize a caller-supplied backend name, without guessing when empty."""
    if not backend or not backend.strip():
        return None
    key = backend.strip().lower()
    return _BACKEND_ALIASES.get(key, key)


def resolve_backend(
    backend: str | None,
    model: str | None,
    default: str = BACKEND_CHATTERBOX,
) -> str:
    """
    Decide which backend owns a request.

    An explicit `backend` always wins. Otherwise a model key unique to a
    backend selects it (`lfm3`), then the configured default applies.
    """
    explicit = normalize_backend(backend)
    if explicit is not None:
        return explicit if explicit in BACKEND_SPECS else default

    model_key = (model or "").strip().lower()
    for key, spec in BACKEND_SPECS.items():
        if model_key in spec.model_specs:
            return key
    if model_key in _BACKEND_ALIASES:
        alias = _BACKEND_ALIASES[model_key]
        if alias in BACKEND_SPECS:
            return alias
    return default if default in BACKEND_SPECS else BACKEND_CHATTERBOX


def get_spec(model_key: str | None, backend: str | None = None):
    """
    Look up a model spec, optionally within one backend.

    Each backend owns its aliases (`multi` for Chatterbox, `omni` for LFM3), so
    resolution is delegated to the backend rather than duplicated here. Without
    an explicit backend, Chatterbox is tried first to preserve the historical
    meaning of short keys like `nano`.
    """
    if not model_key:
        return None
    key = model_key.strip().lower()
    if backend == BACKEND_CHATTERBOX:
        return chatterbox_registry.get_spec(key)
    if backend == BACKEND_LFM3:
        return lfm3_registry.get_spec(key)
    for lookup in (chatterbox_registry.get_spec, lfm3_registry.get_spec):
        found = lookup(key)
        if found is not None:
            return found
    return None


def tags_for(model_key: str, backend: str | None = None) -> frozenset[str]:
    """Bracket tokens the target model actually recognizes."""
    if (backend or resolve_backend(None, model_key)) == BACKEND_CHATTERBOX:
        return chatterbox_registry.tags_for(model_key)
    return frozenset()


def describe_backends() -> list[dict]:
    """JSON-safe backend inventory for tts_status."""
    return [
        {
            "key": spec.key,
            "label": spec.label,
            "default_model": spec.default_model,
            "models": list(spec.model_specs),
            "stock_voice": spec.stock_voice,
            "voice_cloning": spec.cloning,
            "note": spec.note,
        }
        for spec in BACKEND_SPECS.values()
    ]


def describe_models(backend: str | None = None) -> list[dict]:
    """JSON-safe model inventory, tagged with its owning backend."""
    selected = [BACKEND_SPECS[backend]] if backend in BACKEND_SPECS else list(BACKEND_SPECS.values())
    result: list[dict] = []
    for backend_spec in selected:
        for spec in backend_spec.model_specs.values():
            entry = {
                "key": spec.key,
                "backend": backend_spec.key,
                "label": spec.label,
                "parameters": spec.params,
                "languages": spec.languages,
                "stock_voice": spec.stock_voice,
                "requires_reference_clip": not spec.stock_voice,
                "tunable": sorted(k for k in spec.defaults if k in spec.honored_knobs),
                "note": spec.note,
            }
            result.append(entry)
    return result


def load_backend(spec: BackendSpec):
    """Import an engine class lazily: status calls must not load torch."""
    module = importlib.import_module(spec.engine_module)
    return getattr(module, spec.engine_attr)


__all__ = [
    "BACKEND_CHATTERBOX",
    "BACKEND_LFM3",
    "BACKEND_SPECS",
    "SUPPORTED_LANGUAGES",
    "T3_MODELS",
    "VOICE_SUFFIXES",
    "BackendSpec",
    "describe_backends",
    "describe_models",
    "get_spec",
    "load_backend",
    "normalize_backend",
    "resolve_backend",
    "tags_for",
]
