"""Persistent direct SIP client and authenticated Hermes control API."""

import argparse
import asyncio
import hmac
import logging
import os
import shutil
import signal
from pathlib import Path

import aiohttp
from aiohttp import web

from .config import Config
from .conversation import Conversations
from .sip import SipClient
from .store import ID, Store, new_id, validate_call
from .voice import Voice

log = logging.getLogger(__name__)


class Dialer:
    def __init__(self, config: Config, store: Store, sip: SipClient, conversations: Conversations):
        self.config, self.store, self.sip = config, store, sip
        self.conversations = conversations
        self.tasks: set[asyncio.Task] = set()

    async def dial(self, data: dict) -> dict:
        call = validate_call(data, self.config.allowed_extensions)
        call["request_chat_ref"] = call["chat_ref"]
        # Check idempotency against the original request parameters before binding resolution.
        if self.store.get(call["id"]):
            result, _ = self.store.add(call)
            return result
        call["chat_ref"] = self.store.binding(call["extension"]) or call["chat_ref"]
        # Reserve the single line atomically before asynchronous TTS preparation.
        if not self.sip.registered:
            raise web.HTTPServiceUnavailable(
                text='{"error":"SIP account is not registered"}', content_type="application/json"
            )
        if self.sip.busy:
            raise web.HTTPConflict(
                text='{"error":"telephone is busy"}', content_type="application/json"
            )
        result, created = self.store.add(call)
        assert created
        self.sip.reserved = call["id"]
        task = asyncio.create_task(self.prepare_and_dial(call))
        self.tasks.add(task)
        task.add_done_callback(self.finished)
        return result

    def finished(self, task: asyncio.Task) -> None:
        self.tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("call preparation cleanup failed (%s)", type(task.exception()).__name__)

    async def prepare_and_dial(self, call: dict) -> None:
        submitted = False
        try:
            async with asyncio.timeout(240):
                audio = await self.conversations.prepare_initial(call["id"], call["message"])
            if not self.sip.registered:
                raise ConnectionError("registration lost during preparation")
            self.store.transition(call["id"], "preparing", "dialing")
            submitted = True
            self.sip.dial(call["id"], call["extension"], call["message"], audio, call["chat_ref"])
        except asyncio.CancelledError:
            self.store.transition(call["id"], "preparing", "interrupted", "preparation cancelled")
            raise
        except Exception as exc:
            self.store.transition(
                call["id"],
                "dialing" if submitted else "preparing",
                "unknown" if submitted else "failed",
                "INVITE uncertain; inspect SIP server"
                if submitted
                else "initial message preparation failed",
            )
            log.warning("call preparation/submission failed (%s)", type(exc).__name__)
        finally:
            if self.sip.reserved == call["id"]:
                self.sip.reserved = None

    async def close(self) -> None:
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def application(config: Config, store: Store, dialer: Dialer) -> web.Application:
    @web.middleware
    async def auth(request: web.Request, handler):
        value = request.headers.get("Authorization", "")
        if not hmac.compare_digest(value.encode(), ("Bearer " + config.api_token).encode()):
            raise web.HTTPUnauthorized()
        try:
            return await handler(request)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    async def dial(request: web.Request):
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        result = await dialer.dial(data)
        return web.json_response(
            result, status=202 if result["state"] in ("preparing", "dialing") else 200
        )

    async def binding(request: web.Request):
        extension = request.match_info["extension"]
        if extension not in (*config.allowed_callers, *config.allowed_extensions):
            raise ValueError("extension must be an allowed phone extension")
        if request.method == "POST":
            data = await request.json()
            if not isinstance(data, dict) or set(data) != {"chat_ref"}:
                raise ValueError("chat_ref is required")
            return web.json_response(store.bind(extension, data["chat_ref"]))
        if request.method == "DELETE":
            store.unbind(extension)
        return web.json_response({"extension": extension, "chat_ref": store.binding(extension)})

    async def calls(request: web.Request):
        return web.json_response({"calls": store.list()})

    async def health(request: web.Request):
        sip = dialer.sip
        return web.json_response(
            {
                "sip_registered": sip.registered,
                "sip_status": sip.registration_status,
                "busy": sip.busy,
            },
            status=200 if sip.registered else 503,
        )

    app = web.Application(middlewares=[auth], client_max_size=16384)
    app.add_routes(
        [
            web.post("/calls", dial),
            web.get("/calls", calls),
            web.get("/health", health),
            web.get("/bindings/{extension}", binding),
            web.post("/bindings/{extension}", binding),
            web.delete("/bindings/{extension}", binding),
        ]
    )
    return app


async def serve(config: Config) -> None:
    import fcntl

    config.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (config.data_dir / "service.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        media = config.data_dir / "media"
        media.mkdir(mode=0o700, exist_ok=True)
        for item in media.iterdir():
            if item.is_dir() and ID.fullmatch(item.name):
                shutil.rmtree(item)
        store = Store(config.data_dir / "calls.sqlite")
        store.recover()
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as http:
            conversations = Conversations(config, store, Voice(config, http))
            sip = SipClient(config, conversations.accept)
            conversations.sip = sip
            dialer = Dialer(config, store, sip, conversations)
            runner = web.AppRunner(
                application(config, store, dialer),
                access_log=None,
                shutdown_timeout=5,
            )
            stopped = asyncio.Event()
            loop = asyncio.get_running_loop()
            tasks = []
            try:
                sip.inbound_audio = await conversations.prepare_initial(
                    new_id(), config.incoming_message
                )
                sip.start()
                await runner.setup()
                await web.TCPSite(runner, "127.0.0.1", config.api_port).start()
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(sig, stopped.set)
                tasks = [asyncio.create_task(sip.poll()), asyncio.create_task(stopped.wait())]
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    await task  # A failed SIP event loop must stop/restart the service.
            finally:
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.remove_signal_handler(sig)
                await runner.cleanup()
                await dialer.close()
                await conversations.close()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await sip.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path.home() / ".config/hermes-call/config.json"
    )
    parser.add_argument(
        "--check", action="store_true", help="validate config and native SIP binding"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = Config.load(args.config)
    if args.check:
        import importlib

        importlib.import_module("pjsua2")
        print("Configuration and PJSUA2 import valid (network registration not checked)")
        return
    os.umask(0o077)
    asyncio.run(serve(config))


if __name__ == "__main__":
    main()
