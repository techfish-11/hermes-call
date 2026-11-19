import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_call.store import Store, validate_call


def payload(**kwargs):
    return {"extension": "1001", "message": "hello", "request_id": "a" * 32, **kwargs}


@pytest.mark.parametrize(
    "change",
    [
        {"extension": "9999"},
        {"extension": "1001@evil"},
        {"extension": 1001},
        {"message": " "},
        {"message": "a" * 2001},
        {"request_id": "../bad"},
        {"delay_seconds": 1},
        {"at": "2027-01-01"},
    ],
)
def test_reject_untrusted_fields(change):
    with pytest.raises(ValueError):
        validate_call(payload(**change), ("1001",))


def test_atomic_idempotency_and_state_recovery(tmp_path):
    store = Store(tmp_path / "calls.sqlite")
    call = validate_call(payload(), ("1001",))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: store.add(call), range(8)))
    assert sum(created for _, created in results) == 1
    assert len(store.list()) == 1
    with pytest.raises(ValueError, match="different"):
        store.add({**call, "message": "changed"})
    assert store.transition(call["id"], "preparing", "dialing")
    assert store.transition(call["id"], "dialing", "active")
    assert not store.transition(call["id"], "dialing", "active")
    store.recover()
    assert store.get(call["id"])["state"] == "unknown"
    _, created = store.add(call)
    assert not created  # Even ambiguous calls cannot be accidentally redialed.


def test_migration_preserves_old_calls_and_durable_links(tmp_path):
    path = tmp_path / "old.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE calls (id TEXT PRIMARY KEY, extension TEXT,
            message TEXT, created REAL, updated REAL, state TEXT, detail TEXT)""")
        db.execute(
            "INSERT INTO calls VALUES (?,?,?,?,?,?,?)",
            ("a" * 32, "1001", "old message", 1, 1, "completed", ""),
        )
    store = Store(path)
    assert store.get("a" * 32)["message"] == "old message"
    assert store.get("a" * 32)["chat_ref"] is None
    store.bind("1001", "chat-original")
    assert Store(path).binding("1001") == "chat-original"
    store.unbind("1001")
    assert store.binding("1001") is None


@pytest.mark.parametrize("reference", ["latest", "", "-q", "chat\nother", "a" * 201, 123])
def test_reject_ambiguous_or_invalid_chat_references(tmp_path, reference):
    store = Store(tmp_path / "calls.sqlite")
    with pytest.raises(ValueError):
        store.bind("1001", reference)


def test_insert_uses_explicit_columns_regardless_of_table_column_order(tmp_path):
    path = tmp_path / "reordered.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE calls (id TEXT PRIMARY KEY, chat_ref TEXT,
            extension TEXT NOT NULL, message TEXT NOT NULL,
            created REAL NOT NULL, updated REAL NOT NULL, state TEXT NOT NULL,
            detail TEXT NOT NULL DEFAULT '')""")
    store = Store(path)
    added, created = store.add(
        {"id": "b" * 32, "extension": "1001", "message": "hello", "chat_ref": "chat-1"}
    )
    assert created
    assert added["extension"] == "1001"
    assert added["chat_ref"] == "chat-1"
    assert added["state"] == "preparing"
    assert "request_chat_ref" not in added
    assert "request_chat_ref" not in store.get("b" * 32)
    assert "request_chat_ref" not in store.list()[0]
    with pytest.raises(ValueError, match="different"):
        store.add({"id": "b" * 32, "extension": "1001", "message": "hello", "chat_ref": "chat-2"})


def test_migration_preserves_retry_compatibility_for_legacy_bound_calls(tmp_path):
    path = tmp_path / "legacy_bound.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE calls (
            id TEXT PRIMARY KEY, extension TEXT NOT NULL, message TEXT NOT NULL,
            created REAL NOT NULL, updated REAL NOT NULL,
            state TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '', chat_ref TEXT)""")
        # Legacy call created while extension was bound to "bound-chat" (original chat_ref unknown).
        db.execute(
            "INSERT INTO calls VALUES (?,?,?,?,?,?,?,?)",
            ("a" * 32, "1001", "hello", 1, 1, "completed", "", "bound-chat"),
        )
        # Legacy call created with no binding and no originating chat_ref.
        db.execute(
            "INSERT INTO calls VALUES (?,?,?,?,?,?,?,?)",
            ("b" * 32, "1001", "hello", 2, 2, "completed", "", None),
        )

    store = Store(path)
    # Retrying the legacy bound call with omitted or originating chat_ref must still succeed.
    for retry_ref in (None, "origin-chat", "bound-chat"):
        row, created = store.add(
            {
                "id": "a" * 32,
                "extension": "1001",
                "message": "hello",
                "chat_ref": retry_ref,
                "request_chat_ref": retry_ref,
            }
        )
        assert not created
        assert row["chat_ref"] == "bound-chat"
        assert "request_chat_ref" not in row

    # Changing message on a legacy call is still rejected.
    with pytest.raises(ValueError, match="different"):
        store.add(
            {
                "id": "a" * 32,
                "extension": "1001",
                "message": "changed",
                "chat_ref": None,
                "request_chat_ref": None,
            }
        )

    # Legacy call with chat_ref=NULL definitely had no originating chat_ref.
    row, created = store.add(
        {
            "id": "b" * 32,
            "extension": "1001",
            "message": "hello",
            "chat_ref": None,
            "request_chat_ref": None,
        }
    )
    assert not created
    assert row["chat_ref"] is None
    with pytest.raises(ValueError, match="different"):
        store.add(
            {
                "id": "b" * 32,
                "extension": "1001",
                "message": "hello",
                "chat_ref": "other-chat",
                "request_chat_ref": "other-chat",
            }
        )
