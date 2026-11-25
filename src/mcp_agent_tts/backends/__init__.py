"""Synthesis backends and the static catalogue that selects between them."""

from __future__ import annotations

from mcp_agent_tts.backends.base import Backend
from mcp_agent_tts.backends.registry import (
    BACKEND_SPECS,
    describe_backends,
    describe_models,
    load_backend,
    resolve_backend,
)

__all__ = [
    "BACKEND_SPECS",
    "Backend",
    "describe_backends",
    "describe_models",
    "load_backend",
    "resolve_backend",
]
