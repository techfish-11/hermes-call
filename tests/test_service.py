import asyncio
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from hermes_call.service import Dialer, application
from hermes_call.store import Store


class FakeSIP:
    def __init__(self):
        self.registered = True
        self.registration_status = 200
        self.busy = False
        self.reserved = None
        self.requests = []
        self.fail = False

    def dial(self, *args):
        self.requests.append(args)
        self.busy = True
        if self.fail:
            raise RuntimeError("native error")


def test_api_auth_and_idempotent_direct_dial(config, tmp_path):
    async def scenario():
        store = Store(tmp_path / "calls.sqlite")
        sip = FakeSIP()
        preparer = AsyncMock()
        preparer.prepare_initial.return_value = b"\0\0"
        dialer = Dialer(config, store, sip, preparer)
        async with TestClient(TestServer(application(config, store, dialer))) as client:
            headers = {"Authorization": "Bearer " + config.api_token}
            assert (await client.get("/calls")).status == 401
            assert (await client.get("/health", headers=headers)).status == 200
            data = {"extension": "1001", "message": "hello", "request_id": "b" * 32}
            assert (await client.post("/calls", headers=headers, json=data)).status == 202
            assert (await client.post("/calls", headers=headers, json=data)).status == 202
            await asyncio.gather(*list(dialer.tasks))
            assert sip.requests == [("b" * 32, "1001", "hello", b"\0\0", None)]
            assert (
                await client.post("/calls", headers=headers, json={**data, "request_id": "c" * 32})
            ).status == 409
            for invalid in ([], {**data, "extension": "9999"}, {**data, "delay_seconds": 10}):
                assert (await client.post("/calls", headers=headers, json=invalid)).status == 400
            assert (await client.post("/calls", headers=headers, data="{")).status == 400
            sip.registered = False
            assert (await client.get("/health", headers=headers)).status == 503
        with pytest.raises(web.HTTPServiceUnavailable):
            await dialer.dial({**data, "request_id": "d" * 32})
        assert len(sip.requests) == 1

    asyncio.run(scenario())


def test_uncertain_native_submission_not_retried(config, tmp_path):
    async def scenario():
        sip = FakeSIP()
        sip.fail = True
        preparer = AsyncMock()
        preparer.prepare_initial.return_value = b"\0\0"
        dialer = Dialer(config, Store(tmp_path / "calls.sqlite"), sip, preparer)
        data = {"extension": "1001", "message": "hello", "request_id": "d" * 32}
        assert (await dialer.dial(data))["state"] == "preparing"
        await asyncio.gather(*list(dialer.tasks))
        assert (await dialer.dial(data))["state"] == "unknown"
        assert len(sip.requests) == 1

    asyncio.run(scenario())


def test_preparation_failure_never_calls_and_releases_line(config, tmp_path):
    async def scenario():
        sip = FakeSIP()
        preparer = AsyncMock()
        preparer.prepare_initial.side_effect = RuntimeError("TTS unavailable")
        store = Store(tmp_path / "calls.sqlite")
        dialer = Dialer(config, store, sip, preparer)
        data = {"extension": "1001", "message": "hello", "request_id": "e" * 32}
        await dialer.dial(data)
        await asyncio.gather(*list(dialer.tasks))
        assert store.get("e" * 32)["state"] == "failed"
        assert sip.requests == []
        assert sip.reserved is None

    asyncio.run(scenario())


def test_chat_binding_api_and_outbound_chat_selection(config, tmp_path):
    async def scenario():
        store = Store(tmp_path / "calls.sqlite")
        sip = FakeSIP()
        preparer = AsyncMock()
        preparer.prepare_initial.return_value = b"\0\0"
        dialer = Dialer(config, store, sip, preparer)
        headers = {"Authorization": "Bearer " + config.api_token}
        async with TestClient(TestServer(application(config, store, dialer))) as client:
            path = "/bindings/1001"
            assert (await client.post(path, json={"chat_ref": "chat-main"})).status == 401
            response = await client.post(path, headers=headers, json={"chat_ref": "chat-main"})
            assert response.status == 200
            assert (await (await client.get(path, headers=headers)).json())[
                "chat_ref"
            ] == "chat-main"
            assert (
                await client.post("/bindings/9999", headers=headers, json={"chat_ref": "chat-main"})
            ).status == 400
            data = {
                "extension": "1001",
                "message": "hello",
                "request_id": "f" * 32,
                "chat_ref": "cron-session",
            }
            await dialer.dial(data)
            await asyncio.gather(*list(dialer.tasks))
            assert sip.requests[0][-1] == "chat-main"
            assert store.get("f" * 32)["chat_ref"] == "chat-main"
            assert (await client.delete(path, headers=headers)).status == 200
            assert store.binding("1001") is None
            # Retrying the same request_id after unbinding/rebinding stays idempotent.
            retried = await dialer.dial(data)
            assert retried["id"] == "f" * 32
            assert retried["chat_ref"] == "chat-main"
            store.bind("1001", "chat-other")
            assert (await dialer.dial(data))["chat_ref"] == "chat-main"
            with pytest.raises(ValueError, match="different"):
                await dialer.dial({**data, "chat_ref": "other-origin"})
            with pytest.raises(ValueError, match="different"):
                await dialer.dial({**data, "message": "changed"})

            # Unbound call must also reject retries with a different explicit chat_ref.
            sip.busy = False
            store.unbind("1001")
            unbound = {
                "extension": "1001",
                "message": "hello",
                "request_id": "1" * 32,
                "chat_ref": "chat-a",
            }
            await dialer.dial(unbound)
            await asyncio.gather(*list(dialer.tasks))
            assert (await dialer.dial(unbound))["chat_ref"] == "chat-a"
            with pytest.raises(ValueError, match="different"):
                await dialer.dial({**unbound, "chat_ref": "chat-b"})
            with pytest.raises(ValueError, match="different"):
                await dialer.dial({**unbound, "chat_ref": None})
        await dialer.close()

    asyncio.run(scenario())


def test_legacy_migrated_bound_call_retry_succeeds_via_dialer(config, tmp_path):
    import sqlite3

    async def scenario():
        path = tmp_path / "legacy.sqlite"
        with sqlite3.connect(path) as db:
            db.execute("""CREATE TABLE calls (
                id TEXT PRIMARY KEY, extension TEXT NOT NULL, message TEXT NOT NULL,
                created REAL NOT NULL, updated REAL NOT NULL,
                state TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '', chat_ref TEXT)""")
            db.execute(
                "INSERT INTO calls VALUES (?,?,?,?,?,?,?,?)",
                ("9" * 32, "1001", "hello", 1, 1, "completed", "", "bound-chat"),
            )
        store = Store(path)
        sip = FakeSIP()
        dialer = Dialer(config, store, sip, AsyncMock())
        # Retry with omitted chat_ref (as in a cron or unbound retry) must return the existing row.
        retried = await dialer.dial(
            {"extension": "1001", "message": "hello", "request_id": "9" * 32}
        )
        assert retried["id"] == "9" * 32
        assert retried["chat_ref"] == "bound-chat"
        assert sip.requests == []

    asyncio.run(scenario())
