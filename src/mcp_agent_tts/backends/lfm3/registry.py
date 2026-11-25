"""
lfm3 registry — the one model and its fixed controls
====================================================
LFM3-Omni-2B speaks with one of two built-in voices. There is no documented
reference-audio voice-cloning hook, so this registry deliberately exposes
`female` and `male` only. Speech languages come from the official model card.
"""

from __future__ import annotations

from mcp_agent_tts.backends.chatterbox.registry import ModelSpec

#: Speech locales from the LFM3-Omni-2B model card. Codes are the ISO 639-1
#: spellings llama.cpp's /tts endpoint expects; both English locales map to en.
SPEECH_LANGUAGES: dict[str, str] = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English (American or British)",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "es": "Spanish",
    "sv": "Swedish",
    "tr": "Turkish",
    "zh": "Chinese",
}

VOICES: tuple[str, ...] = ("female", "male")

MODEL_SPECS: dict[str, ModelSpec] = {
    "lfm3": ModelSpec(
        key="lfm3",
        label="LFM3-Omni-2B",
        module="mcp_agent_tts.backends.lfm3.engine",
        attr="Lfm3Engine",
        params="2.17B",
        languages="18 speech languages (incl. de, fr, es, zh)",
        stock_voice=True,
        honored_knobs=frozenset({"voice", "language"}),
        defaults={"voice": "male", "language": "en"},
        note=(
            "Liquid AI's omni model, rendered through llama.cpp's resident "
            "/tts endpoint. Fixed female/male voices; no voice cloning."
        ),
    ),
}

MODEL_ALIASES = {
    "lfm3-omni": "lfm3",
    "lfm3-omni-2b": "lfm3",
    "omni": "lfm3",
    "liquid": "lfm3",
}


def get_spec(model_key: str | None):
    """Look up the LFM3 model spec, tolerating obvious aliases."""
    if not model_key:
        return None
    key = model_key.strip().lower()
    return MODEL_SPECS.get(MODEL_ALIASES.get(key, key))


def normalize_voice(voice: str | None) -> str | None:
    if voice is None:
        return None
    value = voice.strip().lower()
    return value or None


def describe_models() -> list[dict]:
    return [
        {
            "key": spec.key,
            "label": spec.label,
            "parameters": spec.params,
            "languages": spec.languages,
            "stock_voice": spec.stock_voice,
            "requires_reference_clip": False,
            "voices": list(VOICES),
            "note": spec.note,
        }
        for spec in MODEL_SPECS.values()
    ]


__all__ = [
    "MODEL_ALIASES",
    "MODEL_SPECS",
    "SPEECH_LANGUAGES",
    "VOICES",
    "describe_models",
    "get_spec",
    "normalize_voice",
]
