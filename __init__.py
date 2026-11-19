"""Native Hermes plugin. Stdlib only; the service has its own isolated venv."""

import functools
import json
import logging
import os
import re
import stat
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)


def _request(method: str, endpoint: str, payload: dict | None = None) -> str:
    path = Path(
        os.environ.get(
            "HERMES_CALL_CONFIG",
            str(Path.home() / ".config/hermes-call/config.json"),
        )
    )
    if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError("hermes-call config requires chmod 600")
    config = json.loads(path.read_text(encoding="utf-8"))
    port = config.get("api_port", 8765)
    token = config.get("api_token")
    if type(port) is not int or not 1024 <= port <= 65535 or not isinstance(token, str):
        raise ValueError("invalid hermes-call API configuration")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{endpoint}",
        method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )

    # No proxies or redirects: credentials must never leave loopback.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=70) as response:
        body = response.read(1024 * 1024 + 1)
        if len(body) > 1024 * 1024:
            raise ValueError("hermes-call response too large")
        result = json.loads(body)
        # Dialing confirms PBX submission only. It does not confirm conversation delivery.
        return json.dumps(result, ensure_ascii=False)


def _http_error_message(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read(4096) if getattr(exc, "fp", None) is not None else b""
        if raw:
            body = json.loads(raw)
            detail = body.get("error") if isinstance(body, dict) else None
            if isinstance(detail, str) and 0 < len(detail.strip()) <= 300:
                return detail.strip()
    except Exception:
        pass
    finally:
        exc.close()
    return "telephone service rejected the request"


def _safe_handler(func):
    @functools.wraps(func)
    def wrapped(args: dict, **kwargs) -> str:
        try:
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
            return func(args, **kwargs)
        except urllib.error.HTTPError as exc:
            payload: dict[str, str | int] = {
                "error": _http_error_message(exc),
                "status": exc.code,
            }
            if func.__name__ == "dial" and (exc.code == 409 or exc.code >= 500):
                payload["retry"] = "reuse the same request_id"
            return json.dumps(payload, ensure_ascii=False)
        except ValueError as exc:
            return json.dumps({"error": str(exc)}, ensure_ascii=False)
        except Exception as exc:
            log.error("hermes-call tool failed (%s)", type(exc).__name__)
            retry = (
                "check service; reuse the same request_id"
                if func.__name__ == "dial"
                else "check service"
            )
            return json.dumps(
                {
                    "error": "telephone service unavailable or misconfigured",
                    "retry": retry,
                }
            )

    return wrapped


@_safe_handler
def dial(args: dict, **kwargs) -> str:
    if set(args) != {"extension", "message", "request_id"}:
        raise ValueError("extension, message and request_id are required")
    if not isinstance(args["request_id"], str) or not re.fullmatch(
        r"[a-f0-9]{32}",
        args["request_id"],
    ):
        raise ValueError("request_id must be 32 lowercase hex characters")
    payload = dict(args)
    reference = kwargs.get("conversation_id") or kwargs.get("session_id")
    if reference:
        payload["chat_ref"] = reference
    return _request("POST", "/calls", payload)


@_safe_handler
def link(args: dict, **kwargs) -> str:
    if set(args) != {"extension", "action"} or args["action"] not in ("link", "unlink", "status"):
        raise ValueError("extension and action (link/unlink/status) are required")
    extension = args["extension"]
    if not isinstance(extension, str) or not re.fullmatch(r"[0-9]{1,12}", extension):
        raise ValueError("invalid extension")
    endpoint = "/bindings/" + extension
    if args["action"] == "status":
        return _request("GET", endpoint)
    if args["action"] == "unlink":
        return _request("DELETE", endpoint)
    reference = kwargs.get("conversation_id") or kwargs.get("session_id")
    if not reference:
        raise ValueError("no current chat session; invoke this tool from the chat to link")
    return _request("POST", endpoint, {"chat_ref": reference})


@_safe_handler
def status(args: dict, **kwargs) -> str:
    if args:
        raise ValueError("status takes no arguments")
    return _request("GET", "/calls")


def register(ctx):
    schemas = [
        (
            "hermes_call_dial",
            dial,
            {
                "description": (
                    "Call an allowed SIP extension NOW and converse using Hermes. "
                    "Use only when requested by the user or an authorized cron task. "
                    "For a future call use Hermes cron to invoke this tool later; this tool has NO "
                    "scheduling. A preparing/dialing result does not mean answered or delivered. "
                    "The initial message is mandatory and plays one second after answer. "
                    "Call conversation resumes the bound chat or the originating chat. "
                    "Use a new random request_id for each call and reuse it on retries. "
                    "Never automatically redial an unknown call."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "extension": {"type": "string", "pattern": "^[0-9]{1,12}$"},
                        "message": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 2000,
                            "description": "Required message spoken one second after answer",
                        },
                        "request_id": {
                            "type": "string",
                            "pattern": "^[a-f0-9]{32}$",
                            "description": "Random 32 lowercase hex characters; reuse on retry",
                        },
                    },
                    "required": ["extension", "message", "request_id"],
                },
            },
        ),
        (
            "hermes_call_link",
            link,
            {
                "description": (
                    "Link an allowed phone extension to THIS chat only when the user asks. "
                    "Incoming and outgoing calls then continue this chat's history and memory. "
                    "Use unlink for a dedicated persistent phone session, or status to read. "
                    "The chat ID comes from runtime context; never invent another session ID."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "extension": {"type": "string", "pattern": "^[0-9]{1,12}$"},
                        "action": {"type": "string", "enum": ["link", "unlink", "status"]},
                    },
                    "required": ["extension", "action"],
                },
            },
        ),
        (
            "hermes_call_status",
            status,
            {
                "description": "Read the last 100 call states; unknown needs PBX inspection.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        ),
    ]
    for name, handler, schema in schemas:
        ctx.register_tool(
            name=name, toolset="hermes_call", schema={"name": name, **schema}, handler=handler
        )
