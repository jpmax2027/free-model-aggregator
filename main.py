#!/usr/bin/env python3
"""free-model-aggregator: one server, two wires, free models only.

Exposes an OpenAI-compatible API (POST /v1/chat/completions) and an
Anthropic-compatible API (POST /v1/messages) on a single HTTP server.
Every request is served by a free OpenRouter model, chosen
deterministically from the vendored catalog (catalog.json, ranked at
build time -- see tools/build_catalog.py). Upstream calls always use the
OpenAI chat-completions wire format against the model gateway.

Configuration is read from the environment only:

    OPENROUTER_API_KEY    API key for the gateway (sent as Bearer token)
    MODEL_GATEWAY_URL     gateway base URL (first choice)
    OPENROUTER_BASE_URL   gateway base URL (fallback choice)
                          default: https://openrouter.ai/api/v1
    HOST                  bind address (default 0.0.0.0)
    PORT                  bind port (default 8000)

Standard library only. No wall clock and no unseeded randomness feed any
decision: model order, failover order and generated ids all derive from
the catalog order and a per-process counter.

Usage:
    python3 main.py            run the server
    python3 main.py --list     print the registered catalog document
"""

import argparse
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(HERE, "catalog.json")
DEFAULT_GATEWAY = "https://openrouter.ai/api/v1"
UPSTREAM_TIMEOUT = 60  # seconds per attempt; a timeout triggers failover
MAX_BODY = 32 * 1024 * 1024

# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------


def load_catalog():
    with open(CATALOG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def catalog_document_bytes():
    """The registered catalog document, byte-for-byte as --list prints it."""
    with open(CATALOG_PATH, "rb") as fh:
        return fh.read()


CATALOG = load_catalog()
MODELS = CATALOG["data"]
MODEL_IDS = [m["id"] for m in MODELS]
MODEL_SET = set(MODEL_IDS)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def gateway_base():
    for var in ("MODEL_GATEWAY_URL", "OPENROUTER_BASE_URL"):
        value = os.environ.get(var)
        if value:
            return value.rstrip("/")
    return DEFAULT_GATEWAY


def gateway_key():
    return os.environ.get("OPENROUTER_API_KEY", "")


# --------------------------------------------------------------------------
# Deterministic id generation (counter based, no randomness)
# --------------------------------------------------------------------------

_id_lock = threading.Lock()
_id_counter = [0]


def next_id(prefix):
    with _id_lock:
        _id_counter[0] += 1
        return "%s_%012d" % (prefix, _id_counter[0])


# --------------------------------------------------------------------------
# Model selection
# --------------------------------------------------------------------------


def candidate_models(requested=None, need_tools=False, need_json=False):
    """Ranked free models to try, in deterministic failover order.

    A client-requested model is honoured first only if it is itself a
    catalog (free) model. Feature requirements filter the ranked list to
    models known to support them; if that filter would empty the list,
    the full ranked list is used (better to try than to refuse).
    """
    ordered = list(MODELS)

    def supports(model):
        if need_tools and not model["supports_tools"]:
            return False
        if need_json and not (
            model["supports_response_format"] or model["supports_structured_outputs"]
        ):
            return False
        return True

    filtered = [m["id"] for m in ordered if supports(m)]
    if filtered:
        ordered_ids = filtered
    else:
        ordered_ids = [m["id"] for m in ordered]
    if requested and requested in MODEL_SET:
        ordered_ids = [requested] + [m for m in ordered_ids if m != requested]
    return ordered_ids


# --------------------------------------------------------------------------
# Upstream (gateway) access -- OpenAI wire format only
# --------------------------------------------------------------------------


class UpstreamError(Exception):
    def __init__(self, status, body):
        super().__init__("upstream error %s" % status)
        self.status = status
        self.body = body  # bytes


def _gateway_headers():
    headers = {"Content-Type": "application/json"}
    key = gateway_key()
    if key:
        headers["Authorization"] = "Bearer " + key
    return headers


def _is_retryable(status):
    return status == 429 or 500 <= status <= 599


def call_upstream(payload, candidates):
    """Non-streaming chat completion with deterministic failover.

    Tries each candidate model in order. Provider rejections (429 / 5xx)
    and transport failures (timeout, connection error) move to the next
    candidate; any other HTTP error is forwarded at once. If every
    candidate fails, the last error is raised.
    """
    url = gateway_base() + "/chat/completions"
    last_error = None
    for model_id in candidates:
        attempt = dict(payload)
        attempt["model"] = model_id
        attempt["stream"] = False
        data = json.dumps(attempt).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers=_gateway_headers(), method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT) as resp:
                return model_id, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read() if exc.fp else b""
            if _is_retryable(exc.code):
                last_error = UpstreamError(exc.code, body)
                continue
            raise UpstreamError(exc.code, body)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = UpstreamError(502, json.dumps(
                {"error": {"message": "gateway unreachable: %s" % exc}}
            ).encode("utf-8"))
            continue
    if last_error is not None:
        raise last_error
    raise UpstreamError(502, b'{"error":{"message":"no candidate models"}}')


def open_upstream_stream(payload, candidates):
    """Open a streaming chat completion with failover before first byte.

    Returns (model_id, response) where response is the open HTTP
    response whose body is an SSE stream. Once bytes flow, failover is
    no longer possible; errors before the stream opens fail over.
    """
    url = gateway_base() + "/chat/completions"
    last_error = None
    for model_id in candidates:
        attempt = dict(payload)
        attempt["model"] = model_id
        attempt["stream"] = True
        data = json.dumps(attempt).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers=_gateway_headers(), method="POST"
        )
        try:
            resp = urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT)
            return model_id, resp
        except urllib.error.HTTPError as exc:
            body = exc.read() if exc.fp else b""
            if _is_retryable(exc.code):
                last_error = UpstreamError(exc.code, body)
                continue
            raise UpstreamError(exc.code, body)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = UpstreamError(502, json.dumps(
                {"error": {"message": "gateway unreachable: %s" % exc}}
            ).encode("utf-8"))
            continue
    if last_error is not None:
        raise last_error
    raise UpstreamError(502, b'{"error":{"message":"no candidate models"}}')


def iter_sse_payloads(resp):
    """Yield parsed JSON payloads from an OpenAI-style SSE stream."""
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except ValueError:
            continue


# --------------------------------------------------------------------------
# Anthropic <-> OpenAI translation
# --------------------------------------------------------------------------

FINISH_TO_ANTHROPIC_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


def anthropic_system_text(system):
    """Flatten an Anthropic `system` field (string or blocks) to text."""
    if system is None:
        return None
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else None
    return None


def _anthropic_block_to_openai_parts(block):
    """Translate one Anthropic content block.

    Returns ("parts", [openai content parts]) for ordinary content,
    ("tool_call", openai tool_call) for assistant tool_use, or
    ("tool_message", openai message) for user tool_result.
    """
    btype = block.get("type")
    if btype == "text":
        return ("parts", [{"type": "text", "text": block.get("text", "")}])
    if btype == "image":
        source = block.get("source") or {}
        if source.get("type") == "base64":
            url = "data:%s;base64,%s" % (
                source.get("media_type", "image/png"),
                source.get("data", ""),
            )
        else:
            url = source.get("url", "")
        return ("parts", [{"type": "image_url", "image_url": {"url": url}}])
    if btype == "tool_use":
        return (
            "tool_call",
            {
                "id": block.get("id") or next_id("toolu"),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input") or {}),
                },
            },
        )
    if btype == "tool_result":
        content = block.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            texts = []
            for sub in content:
                if isinstance(sub, dict) and sub.get("type") == "text":
                    texts.append(sub.get("text", ""))
            text = "\n".join(texts)
        else:
            text = ""
        return (
            "tool_message",
            {
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
                "content": text,
            },
        )
    # thinking / redacted_thinking / unknown blocks carry no portable
    # content for the upstream wire; skip them.
    return ("parts", [])


def anthropic_messages_to_openai(messages):
    out = []
    for message in messages or []:
        role = message.get("role", "user")
        content = message.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        parts = []
        tool_calls = []
        tool_messages = []
        for block in content or []:
            if not isinstance(block, dict):
                continue
            kind, value = _anthropic_block_to_openai_parts(block)
            if kind == "parts":
                parts.extend(value)
            elif kind == "tool_call":
                tool_calls.append(value)
            elif kind == "tool_message":
                tool_messages.append(value)
        if parts or tool_calls:
            msg = {"role": role}
            if len(parts) == 1 and parts[0].get("type") == "text":
                msg["content"] = parts[0]["text"]
            elif parts:
                msg["content"] = parts
            else:
                msg["content"] = None
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
        out.extend(tool_messages)
    return out


def anthropic_tools_to_openai(tools):
    out = []
    for tool in tools or []:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema") or {"type": "object"},
                },
            }
        )
    return out


def anthropic_tool_choice_to_openai(choice):
    if not isinstance(choice, dict):
        return None
    ctype = choice.get("type")
    if ctype == "auto":
        return "auto"
    if ctype == "any":
        return "required"
    if ctype == "none":
        return "none"
    if ctype == "tool":
        return {
            "type": "function",
            "function": {"name": choice.get("name", "")},
        }
    return None


def anthropic_request_to_openai(body):
    """Build the upstream OpenAI payload for an Anthropic /v1/messages body."""
    payload = {}
    messages = []
    system_text = anthropic_system_text(body.get("system"))
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.extend(anthropic_messages_to_openai(body.get("messages")))
    payload["messages"] = messages
    payload["max_tokens"] = body.get("max_tokens", 1024)
    for key in ("temperature", "top_p", "top_k"):
        if key in body:
            payload[key] = body[key]
    if body.get("stop_sequences"):
        payload["stop"] = body["stop_sequences"]
    if body.get("tools"):
        payload["tools"] = anthropic_tools_to_openai(body["tools"])
    tool_choice = anthropic_tool_choice_to_openai(body.get("tool_choice"))
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    return payload


def openai_message_to_anthropic(message, model_id, response_id, usage):
    content = []
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        raw_args = function.get("arguments") or ""
        try:
            parsed = json.loads(raw_args) if raw_args else {}
            if not isinstance(parsed, dict):
                parsed = {"value": parsed}
        except ValueError:
            parsed = {"_unparsed": raw_args}
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id") or next_id("toolu"),
                "name": function.get("name", ""),
                "input": parsed,
            }
        )
    if not content:
        content.append({"type": "text", "text": ""})
    usage = usage or {}
    return {
        "id": response_id,
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": content,
        "stop_reason": None,  # filled by caller from finish_reason
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def openai_response_to_anthropic(result, model_id):
    choices = result.get("choices") or [{}]
    choice = choices[0]
    message = choice.get("message") or {}
    response = openai_message_to_anthropic(
        message, model_id, result.get("id") or next_id("msg"), result.get("usage")
    )
    response["model"] = result.get("model") or model_id
    response["stop_reason"] = FINISH_TO_ANTHROPIC_STOP.get(
        choice.get("finish_reason") or "stop", "end_turn"
    )
    return response


# --------------------------------------------------------------------------
# Error rendering per wire format
# --------------------------------------------------------------------------


def openai_error_body(status, raw_body):
    message = None
    err_type = "api_error"
    if status == 429:
        err_type = "rate_limit_error"
    elif status in (400, 404, 422):
        err_type = "invalid_request_error"
    elif status in (401, 403):
        err_type = "authentication_error"
    try:
        parsed = json.loads(raw_body.decode("utf-8"))
        if isinstance(parsed, dict):
            err = parsed.get("error")
            if isinstance(err, dict) and err.get("message"):
                message = err["message"]
            elif parsed.get("message"):
                message = parsed["message"]
    except (ValueError, UnicodeDecodeError):
        pass
    if not message:
        message = raw_body.decode("utf-8", "replace")[:500] or "upstream error"
    return {"error": {"message": message, "type": err_type, "code": status}}


def anthropic_error_body(status, raw_body):
    inner = openai_error_body(status, raw_body)["error"]
    type_map = {
        "rate_limit_error": "rate_limit_error",
        "invalid_request_error": "invalid_request_error",
        "authentication_error": "authentication_error",
    }
    return {
        "type": "error",
        "error": {
            "type": type_map.get(inner["type"], "api_error"),
            "message": inner["message"],
        },
    }


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------

OPENAI_PASSTHROUGH_KEYS = (
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "tools",
    "tool_choice",
    "response_format",
    "parallel_tool_calls",
)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "FreeModelAggregator/1.0"

    # -- plumbing ------------------------------------------------------
    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, status, obj):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_upstream_error(self, exc, wire):
        if wire == "anthropic":
            self._send_json(exc.status, anthropic_error_body(exc.status, exc.body))
        else:
            self._send_json(exc.status, openai_error_body(exc.status, exc.body))

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ValueError("request body too large")
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _open_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _sse_write(self, data_bytes):
        self.wfile.write(b"%x\r\n" % len(data_bytes))
        self.wfile.write(data_bytes)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _sse_close(self):
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    # -- routing ---------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/v1/models":
            data = [
                {
                    "id": m["id"],
                    "object": "model",
                    "created": 0,
                    "owned_by": m["provider"],
                }
                for m in MODELS
            ]
            self._send_json(200, {"object": "list", "data": data})
        elif path == "/":
            self._send_json(
                200,
                {
                    "object": "status",
                    "status": "ok",
                    "models": len(MODELS),
                    "endpoints": [
                        "/v1/chat/completions",
                        "/v1/messages",
                        "/v1/models",
                    ],
                },
            )
        else:
            self._send_json(
                404,
                {"error": {"message": "not found: %s" % path,
                           "type": "invalid_request_error", "code": 404}},
            )

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            body = self._read_body()
        except (ValueError, UnicodeDecodeError) as exc:
            self._send_json(
                400,
                {"error": {"message": "invalid JSON body: %s" % exc,
                           "type": "invalid_request_error", "code": 400}},
            )
            return
        if path == "/v1/chat/completions":
            self._handle_openai(body)
        elif path == "/v1/messages":
            self._handle_anthropic(body)
        else:
            self._send_json(
                404,
                {"error": {"message": "not found: %s" % path,
                           "type": "invalid_request_error", "code": 404}},
            )

    # -- OpenAI wire -----------------------------------------------------
    def _handle_openai(self, body):
        payload = {"messages": body.get("messages") or []}
        for key in OPENAI_PASSTHROUGH_KEYS:
            if key in body:
                payload[key] = body[key]
        need_tools = bool(body.get("tools"))
        response_format = body.get("response_format") or {}
        need_json = response_format.get("type") in ("json_object", "json_schema")
        candidates = candidate_models(body.get("model"), need_tools, need_json)
        if body.get("stream"):
            self._openai_stream(payload, candidates)
            return
        try:
            _model_id, result = call_upstream(payload, candidates)
        except UpstreamError as exc:
            self._send_upstream_error(exc, "openai")
            return
        self._send_json(200, result)

    def _openai_stream(self, payload, candidates):
        try:
            _model_id, resp = open_upstream_stream(payload, candidates)
        except UpstreamError as exc:
            self._send_upstream_error(exc, "openai")
            return
        self._open_sse()
        try:
            for raw in resp:
                line = raw.strip()
                if line:
                    self._sse_write(line + b"\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            resp.close()
            self._sse_close()

    # -- Anthropic wire --------------------------------------------------
    def _handle_anthropic(self, body):
        payload = anthropic_request_to_openai(body)
        need_tools = bool(body.get("tools"))
        candidates = candidate_models(body.get("model"), need_tools, False)
        if body.get("stream"):
            self._anthropic_stream(payload, candidates)
            return
        try:
            model_id, result = call_upstream(payload, candidates)
        except UpstreamError as exc:
            self._send_upstream_error(exc, "anthropic")
            return
        self._send_json(200, openai_response_to_anthropic(result, model_id))

    def _anthropic_event(self, event, obj):
        data = ("event: %s\ndata: %s\n\n" % (event, json.dumps(obj))).encode("utf-8")
        self._sse_write(data)

    def _anthropic_stream(self, payload, candidates):
        try:
            model_id, resp = open_upstream_stream(payload, candidates)
        except UpstreamError as exc:
            self._send_upstream_error(exc, "anthropic")
            return
        msg_id = next_id("msg")
        self._open_sse()
        started = False
        open_blocks = {}  # block index -> "text" | "tool_use"
        tool_state = {}  # upstream tool index -> block index
        next_block = [0]
        finish_reason = "stop"
        usage = {}

        def ensure_started(model_name):
            nonlocal started
            if not started:
                started = True
                self._anthropic_event(
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": msg_id,
                            "type": "message",
                            "role": "assistant",
                            "model": model_name or model_id,
                            "content": [],
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        },
                    },
                )

        def open_text_block():
            idx = next_block[0]
            next_block[0] += 1
            open_blocks[idx] = "text"
            self._anthropic_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            return idx

        def open_tool_block(call):
            idx = next_block[0]
            next_block[0] += 1
            open_blocks[idx] = "tool_use"
            function = call.get("function") or {}
            self._anthropic_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": call.get("id") or next_id("toolu"),
                        "name": function.get("name", ""),
                        "input": {},
                    },
                },
            )
            return idx

        try:
            for chunk in iter_sse_payloads(resp):
                ensure_started(chunk.get("model"))
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta") or {}
                text = delta.get("content")
                if text:
                    text_idx = None
                    for idx, kind in open_blocks.items():
                        if kind == "text":
                            text_idx = idx
                    if text_idx is None:
                        text_idx = open_text_block()
                    self._anthropic_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": text_idx,
                            "delta": {"type": "text_delta", "text": text},
                        },
                    )
                for call in delta.get("tool_calls") or []:
                    t_index = call.get("index", 0)
                    if t_index not in tool_state:
                        tool_state[t_index] = open_tool_block(call)
                    block_idx = tool_state[t_index]
                    function = call.get("function") or {}
                    arguments = function.get("arguments")
                    if arguments:
                        self._anthropic_event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": block_idx,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": arguments,
                                },
                            },
                        )
        except (BrokenPipeError, ConnectionResetError):
            resp.close()
            return
        finally:
            resp.close()
        ensure_started(model_id)
        for idx in sorted(open_blocks):
            self._anthropic_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": idx},
            )
        self._anthropic_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": FINISH_TO_ANTHROPIC_STOP.get(
                        finish_reason, "end_turn"
                    ),
                    "stop_sequence": None,
                },
                "usage": {"output_tokens": usage.get("completion_tokens", 0)},
            },
        )
        self._anthropic_event("message_stop", {"type": "message_stop"})
        self._sse_close()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description="Free model aggregator server")
    parser.add_argument("--list", action="store_true",
                        help="print the registered catalog document and exit")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args(argv)
    if args.list:
        sys.stdout.write(catalog_document_bytes().decode("utf-8"))
        return 0
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    sys.stderr.write(
        "listening on %s:%d with %d free models via %s\n"
        % (args.host, args.port, len(MODELS), gateway_base())
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
