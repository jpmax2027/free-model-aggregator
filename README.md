# free-model-aggregator

One server, two API wires, free models only.

A zero-dependency Python server that accepts both **OpenAI-style**
(`POST /v1/chat/completions`) and **Anthropic-style**
(`POST /v1/messages`) requests, and serves every one of them with a
**free** OpenRouter model. The server picks the model per request,
translates between the wires, and fails over deterministically when a
provider rejects a call.

## Run

```sh
export OPENROUTER_API_KEY="sk-or-..."        # your gateway key
python3 main.py                              # listens on 0.0.0.0:8000
python3 main.py --list                       # print the model catalog
```

No installation step: Python 3 standard library only
(`http.server` for serving, `urllib` for upstream calls).

## Configuration (environment only)

| Variable            | Meaning                                             | Default                          |
| ------------------- | --------------------------------------------------- | -------------------------------- |
| `OPENROUTER_API_KEY`| Bearer key for the model gateway                    | *(none; header omitted if unset)*|
| `MODEL_GATEWAY_URL` | Gateway base URL (first choice)                     | —                                |
| `OPENROUTER_BASE_URL`| Gateway base URL (fallback choice)                 | `https://openrouter.ai/api/v1`   |
| `HOST` / `PORT`     | Bind address / port (also `--host` / `--port`)      | `0.0.0.0` / `8000`               |

No key is ever hardcoded; the key is read from the environment at
request time.

## Endpoints

- `POST /v1/chat/completions` — OpenAI wire: `messages` (incl. `system`),
  `tools` / `tool_choice`, `response_format` JSON mode, `stream: true`
  (SSE, `data: [DONE]` terminated). Non-streaming responses are the
  upstream OpenAI response verbatim, including `usage`.
- `POST /v1/messages` — Anthropic wire: `system` (string or blocks),
  content blocks (`text`, `image`, `tool_use`, `tool_result`), `tools`
  (translated to OpenAI function calling and back, including `tool_use`
  result blocks), `stream: true` (SSE with `message_start`,
  `content_block_start/delta/stop`, `message_delta`, `message_stop`).
- `GET /v1/models` — the registered catalog in OpenAI list format.
- `GET /` — small status document.

## How models are chosen

`catalog.json` is vendored at build time by `tools/build_catalog.py`
from the public OpenRouter model list, keeping only models whose prompt
**and** completion pricing are `"0"` and whose output modality is text.
Each entry is scored by a deterministic ranking function:

```
score = 100 * provider_tier          (large labs rank above unknown ones)
      + parsed parameter size / 10   (capped; keyword fallback)
      + context length share of 1M   (up to 10 points)
      + feature points               (tools +5, structured outputs +5,
                                      response_format +3)
```

Ties break on model id. The `openrouter/free` meta-router is always
last: it is a valid free endpoint, but it would delegate the very choice
this server exists to make.

Per request, the candidate list is that ranked order, filtered to models
known to support the requested features (tools, JSON mode) when the
request needs them; a client-named model is tried first only if it is
itself in the free catalog. On a provider rejection (HTTP 429 or 5xx) or
a transport failure (timeout, connection error) the server fails over to
the next candidate in order; other HTTP errors are forwarded at once,
and if every candidate fails the last error is forwarded to the client.
Streaming requests fail over only before the first byte — once a stream
is open it is relayed (OpenAI wire) or translated chunk-by-chunk
(Anthropic wire).

## Determinism

No wall clock and no unseeded randomness influence behaviour: model
order, failover order, and generated ids (a per-process counter) are all
deterministic. `python3 main.py --list` prints the registered catalog
document and is byte-for-byte identical across runs (it is exactly the
vendored `catalog.json`).

## Layout

```
main.py                  the server (everything, stdlib only)
catalog.json             vendored free-model catalog (ranked)
tools/build_catalog.py   regenerates catalog.json from OpenRouter
LICENSE                  MIT
```

## License

MIT — see `LICENSE`.
