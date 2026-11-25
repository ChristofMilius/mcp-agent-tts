"""Backend selection and LFM3 adapter behaviour, with no model server."""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from mcp_agent_tts.backends.lfm3.engine import Lfm3Engine
from mcp_agent_tts.backends.registry import (
    BACKEND_CHATTERBOX,
    BACKEND_LFM3,
    get_spec,
    normalize_backend,
    resolve_backend,
    tags_for,
)
from mcp_agent_tts.engine import TtsEngine
from mcp_agent_tts.errors import ToolFault
from mcp_agent_tts.speak import Speaker


class TestBackendResolution:
    def test_explicit_backend_wins_over_model(self):
        assert resolve_backend(BACKEND_LFM3, "nano", BACKEND_CHATTERBOX) == BACKEND_LFM3

    def test_model_can_infer_backend(self):
        assert resolve_backend(None, "lfm3", BACKEND_CHATTERBOX) == BACKEND_LFM3
        assert resolve_backend(None, "turbo", BACKEND_CHATTERBOX) == BACKEND_CHATTERBOX

    def test_default_applies_when_nothing_is_named(self):
        assert resolve_backend(None, None, BACKEND_LFM3) == BACKEND_LFM3

    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("cb", BACKEND_CHATTERBOX),
            ("liquid", BACKEND_LFM3),
            ("LFM3-Omni", BACKEND_LFM3),
        ],
    )
    def test_backend_aliases(self, alias, expected):
        assert normalize_backend(alias) == expected
        assert resolve_backend(alias, None, BACKEND_CHATTERBOX) == expected

    def test_model_aliases_are_resolved_by_owner(self):
        assert get_spec("multi", BACKEND_CHATTERBOX).key == "multilingual"
        assert get_spec("omni", BACKEND_LFM3).key == "lfm3"
        assert get_spec("turbo", BACKEND_LFM3) is None

    def test_lfm3_has_no_paralinguistic_tags(self):
        assert tags_for("lfm3", BACKEND_LFM3) == frozenset()
        assert "[laugh]" in tags_for("nano", BACKEND_CHATTERBOX)


class TestDispatcher:
    def test_routes_by_request_and_forwards_knobs(self, cfg, monkeypatch):
        seen = []

        class Recording:
            def synthesize(self, text, model_key, **knobs):
                seen.append((text, model_key, knobs))
                return {"wav": np.zeros(8), "sample_rate": 24000, "device": "fake"}

        engine = TtsEngine(cfg)
        monkeypatch.setattr(engine, "_engine_for", lambda _name: Recording())

        result = engine.synthesize(
            "Hello.", "lfm3", backend=BACKEND_LFM3, voice="female", language="de"
        )
        assert seen == [
            ("Hello.", "lfm3", {"voice": "female", "language": "de"}),
        ]
        assert result["backend"] == BACKEND_LFM3
        assert result["model"] == "lfm3"

    def test_status_invents_no_backends(self, cfg, monkeypatch):
        engine = TtsEngine(cfg)
        monkeypatch.setattr(
            engine,
            "_engine_for",
            lambda name: type(
                "Fake",
                (),
                {"status": lambda self: {"backend": name, "loaded_model": None}},
            )(),
        )
        status = engine.status()
        assert [item["key"] for item in status["backends"]] == [
            BACKEND_CHATTERBOX,
            BACKEND_LFM3,
        ]
        assert set(status["engines"]) == {BACKEND_CHATTERBOX, BACKEND_LFM3}

    def test_is_loaded_falls_back_to_status_result(self, cfg):
        engine = TtsEngine(cfg)
        engine._last_backend = BACKEND_LFM3
        engine._engines[BACKEND_LFM3] = type(
            "Fake", (), {"status": lambda self: {"loaded_model": "lfm3"}}
        )()
        assert engine.is_loaded() is True
        engine._engines[BACKEND_LFM3] = type("Fake", (), {"status": lambda self: {}})()
        assert engine.is_loaded() is False


class TestLfm3Engine:
    def test_rejects_reference_clip(self, cfg):
        with pytest.raises(ToolFault) as exc:
            Lfm3Engine(cfg).synthesize("Hello.", "lfm3", reference_clip="voice.wav")
        assert exc.value.reason == "voice_cloning_unsupported"

    def test_rejects_unsupported_voice(self, cfg):
        engine = Lfm3Engine(cfg)
        with pytest.raises(ToolFault) as exc:
            engine._voice("narrator")
        assert exc.value.reason == "unsupported_voice"
        assert engine._voice(" FEMALE ") == "female"

    def test_rejects_unsupported_language(self, cfg):
        engine = Lfm3Engine(cfg)
        with pytest.raises(ToolFault) as exc:
            engine._language("xx")
        assert exc.value.reason == "unsupported_language"
        assert engine._language(" DE ") == "de"

    def test_paths_are_required_before_starting_a_server(self, cfg):
        cfg.lfm3_model = ""
        cfg.lfm3_mmproj = ""
        with pytest.raises(ToolFault) as exc:
            Lfm3Engine(cfg)._paths()
        assert exc.value.reason == "lfm3_not_configured"

    def test_decodes_mono_pcm16_wav(self):
        payload = io.BytesIO()
        with wave.open(payload, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(24000)
            writer.writeframes(np.array([0, 16384, -16384], dtype="<i2").tobytes())
        wav, sample_rate = Lfm3Engine._decode_wav(payload.getvalue())
        assert sample_rate == 24000
        assert wav.dtype == np.float32
        assert float(wav[1]) == pytest.approx(0.5)

    def test_synthesize_uses_fixed_voice_without_reference_audio(self, cfg, monkeypatch):
        engine = Lfm3Engine(cfg)
        payload = io.BytesIO()
        with wave.open(payload, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(24000)
            writer.writeframes(np.zeros(240, dtype="<i2").tobytes())

        monkeypatch.setattr(engine, "_ensure_server", lambda: None)
        monkeypatch.setattr(engine, "_post_tts", lambda text, voice, language: payload.getvalue())
        result = engine.synthesize("Hello.", "lfm3", voice="female", language="de")
        assert result["voice"] == "female"
        assert result["language"] == "de"
        assert result["backend"] == BACKEND_LFM3
        assert result["reference_clip_used"] is False
        assert result["sample_rate"] == 24000


class TestSpeakerBackendValidation:
    def test_lfm3_uses_fixed_voice_and_no_clip(self, cfg, fake_engine):
        cfg.lfm3_voice = "female"
        speaker = Speaker(cfg, fake_engine)
        payload = speaker.submit("Hello.", backend=BACKEND_LFM3, voice="male", language="en")
        speaker.wait()
        speaker.stop_worker()

        assert payload["backend"] == BACKEND_LFM3
        assert payload["model"] == "lfm3"
        assert payload["voice"] == "male"
        call = fake_engine.calls[0]
        assert call["backend"] == BACKEND_LFM3
        assert call["voice"] == "male"
        assert "reference_clip" not in call

    def test_lfm3_rejects_reference_clip_before_queueing(self, cfg, fake_engine):
        speaker = Speaker(cfg, fake_engine)
        with pytest.raises(ToolFault) as exc:
            speaker.submit("Hello.", backend=BACKEND_LFM3, reference_clip="narrator.wav")
        assert exc.value.reason == "voice_cloning_unsupported"
        assert fake_engine.calls == []

    def test_lfm3_rejects_chatterbox_model(self, cfg, fake_engine):
        speaker = Speaker(cfg, fake_engine)
        with pytest.raises(ToolFault) as exc:
            speaker.submit("Hello.", backend=BACKEND_LFM3, model="nano")
        assert exc.value.reason == "unknown_model"
        assert fake_engine.calls == []

    def test_unknown_backend_is_reported(self, cfg, fake_engine):
        speaker = Speaker(cfg, fake_engine)
        with pytest.raises(ToolFault) as exc:
            speaker.submit("Hello.", backend="elevenlabs")
        assert exc.value.reason == "unknown_backend"
        assert fake_engine.calls == []
