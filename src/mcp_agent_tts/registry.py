"""
registry.py — compatibility façade over the backend registry
============================================================
Old call sites (and the ported Chatterbox tests) import model tags and specs
from `mcp_agent_tts.registry`. The catalogue now lives in `backends.registry`;
this module keeps the flat names available while making the backend ownership
explicit to new code.
"""

from __future__ import annotations

from mcp_agent_tts.backends.chatterbox.registry import (
    MODEL_SPECS,
    ORIGINAL_TAGS,
    T3_MODELS,
    TURBO_TAGS,
    VOICE_SUFFIXES,
    ModelSpec,
    load_model_class,
    model_keys,
)
from mcp_agent_tts.backends.chatterbox.registry import (
    SUPPORTED_LANGUAGES as CHATTERBOX_LANGUAGES,
)
from mcp_agent_tts.backends.registry import (
    BACKEND_CHATTERBOX,
    BACKEND_LFM3,
    BACKEND_SPECS,
    SUPPORTED_LANGUAGES,
    BackendSpec,
    describe_backends,
    describe_models,
    get_spec,
    load_backend,
    normalize_backend,
    resolve_backend,
    tags_for,
)

__all__ = [
    "BACKEND_CHATTERBOX",
    "BACKEND_LFM3",
    "BACKEND_SPECS",
    "CHATTERBOX_LANGUAGES",
    "MODEL_SPECS",
    "ORIGINAL_TAGS",
    "SUPPORTED_LANGUAGES",
    "T3_MODELS",
    "TURBO_TAGS",
    "VOICE_SUFFIXES",
    "BackendSpec",
    "ModelSpec",
    "describe_backends",
    "describe_models",
    "get_spec",
    "load_backend",
    "load_model_class",
    "model_keys",
    "normalize_backend",
    "resolve_backend",
    "tags_for",
]
