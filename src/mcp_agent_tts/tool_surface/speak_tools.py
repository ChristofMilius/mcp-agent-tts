"""
speak_tools — the main MCP call
===============================
speak: accept text for speech and hand it to the resident synthesis worker.
       Talks to worker.py over loopback, and answers the moment the worker has
       accepted the text — the reply then plays out in the worker.

The MCP server does NOT synthesize. It used to, and it was unreliable: inside
the server process rendering stopped after two sentences and never resumed,
while the same code driven from a plain script finished every sentence (see
worker.py for the full measurement). The server's job now is to carry text to
the process that does this reliably.

speak() does not wait for the utterance to finish. It used to — the tool
result was an honest, sentence-by-sentence account of what was written and
played — but a blocking result holds the caller's tool call open for the
entire spoken duration, and a harness with a tool-call timeout (LM Studio)
aborts such a call long before a long reply finishes, cutting the turn off
with audio still playing. Returning on acceptance makes the tool independent
of any client timeout: the worker being resident is what keeps the audio
playing regardless of what the harness does with this process. The finished
record is served afterwards by tts_status under worker.last.

Backend selection:
  * backend="chatterbox" (the default) runs the Resemble AI models listed
    below on the local GPU, with reference-clip voice cloning.
  * backend="lfm3" runs Liquid AI's LFM3-Omni-2B through a resident
    llama-server. It has fixed voices female/male and no reference-audio
    cloning. Passing model="lfm3" selects it even without backend=.
  * If backend= is omitted, a backend-unique model= selects the backend;
    otherwise the configured default backend applies.
"""

from __future__ import annotations

import json

from mcp_agent_tts.errors import tool_error
from mcp_agent_tts.worker import WorkerUnavailable, send_request

#: Argument names the worker accepts on a speak request. Used by
#: _worker_request() to copy the caller's arguments across without repeating
#: sixteen keyword names a second time.
_KNOBS = (
    "backend",
    "model",
    "voice",
    "reference_clip",
    "language",
    "t3_model",
    "exaggeration",
    "cfg_weight",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "norm_loudness",
    "seed",
    "play_audio",
    "filename",
)


def _worker_request(cfg, arguments: dict) -> dict:
    """
    Build the worker request, carrying only the knobs the caller actually set.

    An omitted knob must arrive absent rather than as an explicit null: the
    worker reads null as "unset" either way, but keeping the payload to what was
    asked for makes the request readable in the log.
    """
    request = {"op": "speak", "text": arguments["text"]}
    for name in _KNOBS:
        value = arguments.get(name)
        if value is not None:
            request[name] = value
    return request


def register(server, ctx) -> None:
    cfg = ctx.cfg

    @server.tool()
    def speak(
        text: str,
        backend: str | None = None,
        model: str | None = None,
        voice: str | None = None,
        reference_clip: str | None = None,
        language: str | None = None,
        t3_model: str | None = None,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        norm_loudness: bool | None = None,
        seed: int | None = None,
        play_audio: bool | None = None,
        filename: str | None = None,
    ) -> str:
        """
        Speak text aloud with a selected local TTS backend.

        WHEN TO CALL: whenever the user asks for a spoken reply -- "speak",
        "answer aloud", "reply by voice", "say it out loud", and the like --
        put your ENTIRE reply into text= and call this tool ONCE. Do not reply
        in plain text alone while the user is expecting voice; the spoken
        output is the deliverable. text= is YOUR generated reply -- never copy
        the user's message into it; the only exception is when they quoted
        exact words to read aloud. Keep the text in spoken register: full
        sentences, no markdown, no lists, no paths or code. One instruction to
        speak stays in effect for the whole exchange until the user says stop.
        Voice mode ON means EVERY reply is spoken, not just the first.

        TAGS: bracketed tokens act as performance cues ONLY when the target
        model's vocabulary really contains them — assumption is not allowed,
        the spelling must be exact from the list. For nano and turbo (the
        default models) the real token set is these 19: [advertisement],
        [angry], [chuckle], [clear throat], [cough], [crying], [dramatic],
        [fear], [gasp], [groan], [happy], [laugh], [narration], [sarcastic],
        [shush], [sigh], [sniff], [surprised], [whispering]. Watch the two
        easy traps: it is [whispering], never [whisper], and [clear throat]
        has a space, never [clearthroat]. Anything else you bracket — an
        invented tag, or a tag from the 500M models' bigger vocabulary such
        as [humming] or [music] — is dropped from the text before rendering
        and its mood never reaches the voice. These
        tokens are cues, not subtitles: use them sparingly, a few per reply
        at most, and put the rest of the mood into word choice and
        punctuation.

        One call, one reply. Do not call speak() again while the audio for the
        previous call is still playing -- your harness runs one tool call at a
        time, so a second call queues behind the first and would be spoken
        again. Leaving out the tail of a reply ("...") because it seems long is
        a mistake: the text budget is generous and long text is split per
        sentence automatically, so there is nothing to shorten.

        This call returns the moment the resident worker ACCEPTS the text, not
        when the reply finishes playing; synthesis and playback then continue
        in the worker, untouched by any harness tool-call timeout. Do not
        narrate progress and do not poll tts_status() while playback is
        running -- the audio is the deliverable to the user and needs nothing
        from you. Once the utterance has finished, tts_status() carries its
        record (chunk files, played counts, timings, errors) under worker.last.

        After the acceptance result, your harness still wants a final chat
        message: close the turn by writing out the EXACT text you passed into
        text= -- word for word, same punctuation, with the bracketed
        paralinguistic tags (see TAGS above) removed. The chat message is
        the transcript of what was spoken, not a second reply. Do not tidy,
        shorten, rephrase or expand it, and do not append commentary about the
        call: when the written version drifts from the spoken one, the user
        reads and hears two different replies, which is precisely the failure
        this instruction exists to prevent.

        If the reply comes back with reason "worker_not_running", the resident
        synthesis worker is not up: nothing was spoken. Tell the user to start
        it (`mcp-agent-tts worker`) rather than retrying.

        Model and voice are chosen by the user, not by you. If the user has not
        named a voice, ask once which voice to use before the first spoken
        reply, then keep that voice for the rest of the exchange -- unless
        tts_status reports a configured default_voice (use it silently, no
        question), or the user is waiting on the content (speak in the built-in
        voice and offer the voice choice alongside the answer).

        BACKEND: pass this only when the user named an engine, or to try the
        other one on request. Omit it otherwise; the configured default is a
        deliberate local choice. model="lfm3" also selects LFM3.

        MODEL: "nano" is the Chatterbox house default for English (110M, same
        built-in voice and inline tags as turbo, lighter and quicker). Do NOT
        pass model= to "upgrade" it or to express a preference of your own —
        omit it and the worker's configured default stands. Pass model= only
        when the user asked for a specific model, or to experiment when they
        asked you to try one ("use turbo", "try the multilingual one").
        Overriding without being asked is a bug, not an optimisation.

        backend: "chatterbox" (default) or "lfm3". Both can also be reached
                 through their aliases in tts_status.
        model:   omit to use the backend's configured default. Chatterbox:
                 "nano" (110M, built-in voice, inline tags), "turbo" (350M,
                 same features, more VRAM), "multilingual" (500M, 23 languages,
                 needs a clip) or "original" (500M, English, needs a clip).
                 LFM3: "lfm3" only. Pass this only on request — see MODEL above.
        voice:   Chatterbox: name of a reference clip in the voices directory
                 (its filename without extension). Cloning works on every
                 model; turbo/nano fall back to their built-in voice when no
                 clip is given. multilingual/original require a clip.
                 LFM3: fixed "female" or "male", not a clip name.
        reference_clip: Chatterbox only. Path to a wav/mp3/flac clip, as an
                 alternative to voice=. Supply as much clean continuous speech
                 as you have. Short references measurably increase fine clicks
                 and crackle: a 40 s reference beat an 11.8 s excerpt of the
                 same recording by a wide margin, several times the model's own
                 run-to-run variance. Under ~5 s is genuinely too little. Do not
                 normalise or limit the clip -- that shifts output level and
                 adds artefacts. A mastered or compressed recording is fine.
                 Sample rate and channel count need not match anything. LFM3
                 refuses this argument with voice_cloning_unsupported.
        language: ISO 639-1 code. Chatterbox accepts it on "multilingual" only
                 (e.g. "de", "en", "fr"); LFM3 accepts it on its single model.
        t3_model: "v2" (default) or "v3" — multilingual checkpoint.
        temperature: sampling temperature (0.05-5.0). Honored by all models.
        top_p: nucleus-sampling cutoff (0.0-1.0). Honored by all models.
        top_k: top-k sampling size (0-1000). Honored by turbo/nano only — the
                 500M models have no such parameter.
        repetition_penalty: penalise repeated tokens (1.0-2.0). Honored by all
                 models.
        norm_loudness: normalize output to -27 LUFS. Honored by turbo/nano only
                 — the 500M models have no such parameter.
        exaggeration/cfg_weight: style controls (0.0-2.0 / 0.0-1.0). Honored
                 ONLY by the 500M models (multilingual/original); turbo/nano
                 ignore both, so setting them with model="turbo" is dropped
                 with a warning, not an error. cfg_weight>0 doubles the text
                 tokens for CFG guidance.
        seed: reseed torch (CPU + CUDA) so re-renders are reproducible within
                 the resident worker's session; 0 (the upstream convention) or
                 unset keeps random sampling. Byte-identical output across a
                 worker restart is not guaranteed — Chatterbox is
                 nondeterministic across CUDA kernel choices.
                 Left unset, every knob keeps the model's own tuned default
                 (e.g. turbo runs cfg_weight 0.0 with top_k 1000; the 500M
                 models run cfg_weight 0.5).

        play_audio: set false to only write the chunk files. Defaults to the
                 configured setting (on).
        filename: base name for the chunk files; they are written as
                 <filename>_1.wav, <filename>_2.wav, ... A timestamped message
                 id is used when omitted.

        Text is cut into chunks of whole sentences (up to three by default, see
        MCP_TTS_SENTENCES_PER_CHUNK, always capped by MCP_TTS_MAX_CHUNK_CHARS
        characters and by speaking time relative to the chunk before it). Chunks
        are played in order from a small render-ahead buffer, so the reply plays
        continuously from the first sentence instead of pausing for a fresh
        synthesis at every boundary. Files land in the output directory and replay
        in filename order. Returns JSON with the message id, the selected
        backend/model, the chunk count, the file paths the chunks will be written
        to, and how many utterances are queued ahead. The FIRST Chatterbox call
        in a fresh worker downloads weights from Hugging Face (hundreds of MB to
        a few GB); the first LFM3 call starts a resident llama-server and loads
        its GGUF. Later calls reuse both.
        """
        try:
            response = send_request(
                _worker_request(
                    cfg,
                    {
                        "text": text,
                        "backend": backend,
                        "model": model,
                        "voice": voice,
                        "reference_clip": reference_clip,
                        "language": language,
                        "t3_model": t3_model,
                        "exaggeration": exaggeration,
                        "cfg_weight": cfg_weight,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "repetition_penalty": repetition_penalty,
                        "norm_loudness": norm_loudness,
                        "seed": seed,
                        "play_audio": play_audio,
                        "filename": filename,
                    },
                ),
                port=cfg.worker_port,
                timeout=cfg.worker_connect_timeout_s,
            )
        except WorkerUnavailable as exc:
            return json.dumps(
                {
                    "status": "failed",
                    "reason": "worker_not_running",
                    "message": (
                        f"The synthesis worker is not reachable ({exc}). Start it with "
                        "`mcp-agent-tts worker` and leave it running; the server "
                        "does not synthesize on its own."
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        except Exception as e:  # noqa: BLE001 — the caller sees a safe string
            return tool_error("speak", e)

        if not response.get("ok"):
            return json.dumps(
                {
                    "status": "failed",
                    "reason": response.get("reason", "worker_error"),
                    "message": response.get("message", "The worker refused the request."),
                    **{
                        k: v
                        for k, v in response.items()
                        if k not in {"ok", "reason", "message"}
                    },
                },
                indent=2,
                ensure_ascii=False,
            )

        return json.dumps(
            {
                "status": "accepted",
                "message_id": response.get("message_id"),
                "chunks": response.get("chunks"),
                "text_chars": response.get("text_chars"),
                "files": response.get("files", []),
                "play": response.get("play"),
                "model": response.get("model"),
                "voice": response.get("voice"),
                "queued_ahead": response.get("queued_ahead"),
                "output_dir": response.get("output_dir"),
                "note": (
                    "The text is accepted. Synthesis and playback run in the resident "
                    "worker and continue even if this result is dropped or the turn "
                    "ends; tts_status() reports the queue while it runs and the "
                    "finished record under worker.last."
                ),
            },
            indent=2,
            ensure_ascii=False,
        )


__all__ = ["register"]
