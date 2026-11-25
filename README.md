# mcp-agent-tts

General local text-to-speech MCP harness with switchable backends. The speech
pipeline — queueing, sentence chunking, render-ahead buffering, WAV files,
playback, replay, stop, history and the MCP tool surface — lives here once.
Each synthesis engine is a backend module behind a small interface.

Backends shipped in this first version:

- **chatterbox** — Resemble AI's Chatterbox family: `nano`, `turbo`,
  `multilingual`, `original`. Supports reference-clip voice cloning and inline
  paralinguistic tags where the selected model has them.
- **lfm3** — Liquid AI's LFM3-Omni-2B GGUF through a resident `llama-server`
  `/tts` endpoint. Fixed `female` / `male` voices, 24 kHz WAV output, no
  reference-audio cloning.

For whom: anyone running local TTS models who wants one stable agent-facing
tool surface instead of one MCP server per engine.

## Status

The generic harness is based on the validated `mcp-agent-chatterbox` pipeline.
Chatterbox is now a backend; LFM3 is the second backend. Qwen3-TTS can be added
later without touching the queue or playback code.

## Install

Requires `uv` and Python 3.13.

```powershell
uv sync
```

That installs the base harness only. Add the Chatterbox engine when needed:

```powershell
uv sync --extra chatterbox
```

LFM3 needs an installed `llama.cpp` build whose `llama-server` understands
LFM3-Omni GGUF plus its mmproj file. No Python package is installed for it.

Run the server with:

```powershell
uv run python -m mcp_agent_tts
```

Start the resident synthesis worker separately:

```powershell
uv run python -m mcp_agent_tts worker
```

## Tool surface

| Tool | Purpose |
|---|---|
| `speak` | Queue text for rendering and playback. Accepts `backend=`, `model=`, `voice=`, `language=` and the Chatterbox style knobs. |
| `tts_status` | Report configured backends, models, device/VRAM, worker state and LFM3 server state. Loads nothing. |
| `list_voices` | List Chatterbox reference clips, or LFM3's fixed `female` / `male` voices. |
| `tts_unload` | Evict Chatterbox weights and stop a resident LFM3 server. |
| `stop_speech` | Cut off current playback and rendering. |

Examples:

```python
speak("Hello there", backend="chatterbox", model="nano")
speak("Guten Tag", backend="chatterbox", model="multilingual", voice="narrator", language="de")
speak("Hello there", backend="lfm3", voice="male", language="en")
```

If `backend` is omitted, `model="lfm3"` selects LFM3; otherwise the configured
default backend (`chatterbox`) is used.

## Configuration

Settings live in `.env` in the project root. Copy `.env.example` and edit it.
Environment variables win over the file. The prefix is `MCP_TTS_`.

Core settings:

| Variable | Default | Meaning |
|---|---|---|
| `MCP_TTS_BACKEND` | `chatterbox` | Default backend. |
| `MCP_TTS_MODEL` | `nano` | Default model within that backend. |
| `MCP_TTS_VOICE` | empty | Default reference clip (Chatterbox) or LFM3 voice. |
| `MCP_TTS_OUTPUT_DIR` | `tts_output` | Rendered WAV files. |
| `MCP_TTS_VOICES_DIR` | `voices` | Chatterbox reference clips. |
| `MCP_TTS_LOGS_DIR` | `logs` | Worker and server logs. |
| `MCP_TTS_DEVICE` | `auto` | Chatterbox device: `auto`, `cpu`, `cuda`, `cuda:N`. |
| `MCP_TTS_GPU_INDEX` | unset | Pin a CUDA ordinal for Chatterbox. |
| `MCP_TTS_WORKER_PORT` | `8765` | Loopback port for the resident worker. |

LFM3 settings:

| Variable | Default | Meaning |
|---|---|---|
| `MCP_TTS_LFM3_MODEL` | empty | Absolute path to the LFM3 GGUF backbone. |
| `MCP_TTS_LFM3_MMPROJ` | empty | Absolute path to the LFM3 mmproj GGUF. |
| `MCP_TTS_LFM3_EXECUTABLE` | `llama-server` | llama.cpp server binary. |
| `MCP_TTS_LFM3_SERVER_PORT` | `9933` | Loopback port for its HTTP server. |
| `MCP_TTS_LFM3_VOICE` | `male` | `female` or `male`. |
| `MCP_TTS_LFM3_LANGUAGE` | `en` | Speech language code. |
| `MCP_TTS_LFM3_NGL` | `99` | Layers offloaded to GPU. |
| `MCP_TTS_LFM3_CTX` | `4096` | Server context size. |

No absolute path is committed: put machine-specific paths in the ignored `.env`.

## Architecture

```text
src/mcp_agent_tts/
  core pipeline: config.py, worker.py, speak.py, audio.py, playback.py,
                 output.py, names.py, server.py, tool_surface/
  engine.py      backend dispatcher (implements the pipeline's Synthesizer)
  registry.py    unified backend/model catalogue
  voices.py      compatibility shim for the Chatterbox clip scanner
  backends/
    chatterbox/  engine, model registry, reference-clip voices
    lfm3/        resident llama-server /tts adapter, voices, languages
```

The pipeline depends only on a `synthesize()` call returning `wav`,
`sample_rate`, `device` and `elapsed_s`. Backend-specific validation stays in
the tool surface before a job is queued; backend-specific rendering stays in
the backend module. There is no dynamic plugin loader: the backend table is a
plain dict, which is enough until a fourth backend justifies more.

## Tests

```powershell
uv run pytest
```

## License

MIT. See `LICENSE`.
