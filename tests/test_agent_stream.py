import asyncio
import json
import sys

import pytest

from hermes_call.voice import read_agent_events, run_process


def encoded(event):
    return (json.dumps(event, ensure_ascii=False) + "\n").encode()


def test_fragmented_events_forward_only_tool_metadata_and_terminal_speech():
    async def scenario():
        reader = asyncio.StreamReader()
        events = []
        task = asyncio.create_task(read_agent_events(reader, events.append))
        use = encoded(
            {
                "type": "tool_use",
                "name": "web",
                "tool_call_id": "123",
                "input": {"secret": "do not retain"},
            }
        )
        reader.feed_data(use[:10])
        await asyncio.sleep(0)
        assert events == []
        reader.feed_data(use[10:])
        await asyncio.sleep(0)
        assert events == [
            {"type": "tool_use", "name": "web", "tool_call_id": "123", "is_error": False}
        ]
        assert not task.done()
        reader.feed_data(encoded({"type": "text", "text": "do not speak partial text"}))
        reader.feed_data(
            encoded(
                {
                    "type": "tool_result",
                    "name": "web",
                    "tool_call_id": "123",
                    "output": "private output",
                    "is_error": False,
                }
            )
        )
        reader.feed_data(encoded({"type": "result", "exit_code": 0, "text": "調べました"}))
        reader.feed_eof()
        assert (await task).decode() == "調べました"
        assert len(events) == 2
        assert "input" not in events[0] and "output" not in events[1]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "body",
    [
        b"not json\n",
        encoded({"type": "text", "text": "incomplete"}),
        encoded({"type": "result", "exit_code": 1, "text": "failed"}),
        encoded({"type": "result", "exit_code": 0, "text": "a" * 8001}),
    ],
)
def test_invalid_or_incomplete_stream_fails_without_retry(body):
    async def scenario():
        reader = asyncio.StreamReader()
        reader.feed_data(body)
        reader.feed_eof()
        with pytest.raises(RuntimeError):
            await read_agent_events(reader, None)

    asyncio.run(scenario())


def test_real_child_stream_is_decoded_and_exit_status_checked():
    script = (
        "import json; "
        "print(json.dumps({'type':'tool_use','name':'terminal'}),flush=True); "
        "print(json.dumps({'type':'tool_result','name':'terminal','is_error':False}),flush=True); "
        "print(json.dumps({'type':'result','exit_code':0,'text':'done'}),flush=True)"
    )
    events = []
    result = asyncio.run(
        run_process([sys.executable, "-c", script], 5, stream_json=True, on_event=events.append)
    )
    assert result == "done"
    assert [event["type"] for event in events] == ["tool_use", "tool_result"]
    with pytest.raises(RuntimeError, match="exit 3"):
        asyncio.run(
            run_process(
                [sys.executable, "-c", script + ";raise SystemExit(3)"], 5, stream_json=True
            )
        )


def test_large_tool_event_line_above_256kb_is_accepted_within_stream_limit():
    script = (
        "import json; "
        "print(json.dumps({'type':'tool_result','name':'web','output':'x'*350000}),flush=True); "
        "print(json.dumps({'type':'result','exit_code':0,'text':'ok'}),flush=True)"
    )
    events = []
    result = asyncio.run(
        run_process([sys.executable, "-c", script], 5, stream_json=True, on_event=events.append)
    )
    assert result == "ok"
    assert events == [{"type": "tool_result", "name": "web", "is_error": False}]


def test_oversized_single_line_above_stream_limit_raises_runtime_error():
    script = (
        "import json; "
        "print(json.dumps({'type':'tool_result','output':'x'*(2*1024*1024+1024)}),flush=True)"
    )
    with pytest.raises(RuntimeError, match="exceeded limit"):
        asyncio.run(run_process([sys.executable, "-c", script], 5, stream_json=True))
