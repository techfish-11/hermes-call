import asyncio
import struct
import sys
import wave
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from hermes_call.voice import SpeechResponseError, Voice, read_json_object, run_process, split_text


class SpeechResponse:
    def __init__(self, body):
        self.body = body
        self.content = self
        self.status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    async def read(self, size):
        return self.body[:size]

    async def iter_chunked(self, size):
        yield self.body


def test_groq_and_voicevox_contract(config, tmp_path, monkeypatch):
    requests = []

    class HTTP:
        def post(self, url, **kwargs):
            requests.append((url, kwargs))
            if url.endswith("/transcriptions"):
                return SpeechResponse('{"text":"こんにちは"}'.encode())
            if url.endswith("/audio_query"):
                return SpeechResponse(b'{"outputSamplingRate":24000}')
            return SpeechResponse(b"fake WAV")

    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        wav.writeframes(struct.pack("<h", 1000) * 8000)
    ffmpeg = AsyncMock(return_value="")
    monkeypatch.setattr("hermes_call.voice.run_process", ffmpeg)
    voice = Voice(config, HTTP())
    assert asyncio.run(voice.transcribe(source)) == "こんにちは"
    output = tmp_path / "output.wav"
    asyncio.run(voice.synthesize("こんにちは", output))
    assert requests[0][0] == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert requests[0][1]["headers"]["Authorization"] == "Bearer groq-test-key"
    assert requests[1][0] == "https://voicevox-api.swiftlybot.com/v1/audio_query"
    assert requests[1][1]["json"] == {"text": "こんにちは", "speaker": 3}
    assert requests[2][0] == "https://voicevox-api.swiftlybot.com/v1/synthesis"
    assert requests[2][1]["json"] == {"speaker": 3, "audioQuery": {"outputSamplingRate": 24000}}
    assert "params" not in requests[1][1]
    assert all(not kwargs["allow_redirects"] for _, kwargs in requests)
    argv = ffmpeg.call_args.args[0]
    assert argv[argv.index("-ar") + 1] == "16000"
    assert argv[-1] == str(output)
    assert not output.with_suffix(".source").exists()


def test_long_text_is_split_without_loss():
    text = "あ" * 350 + "。" + "い" * 600
    chunks = split_text(text, 500)
    assert "".join(chunks) == text
    assert all(0 < len(chunk) <= 500 for chunk in chunks)


def test_silence_never_sent_to_whisper(config, tmp_path):
    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        wav.writeframes(struct.pack("<h", 0) * 16000)
    assert asyncio.run(Voice(config, None).transcribe(path)) == ""


def test_short_utterance_with_trailing_silence_reaches_whisper(config, tmp_path):
    class HTTP:
        def post(self, url, **kwargs):
            return SpeechResponse('{"text":"はい"}'.encode())

    path = tmp_path / "short.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        # 0.3s at amplitude 250 + 2.0s silence -> whole-file mean energy < 100**2.
        wav.writeframes(struct.pack("<h", 250) * 4800 + b"\0\0" * 32000)
    assert asyncio.run(Voice(config, HTTP()).transcribe(path)) == "はい"


def test_hermes_argv_preserves_call_session_and_cron(config, monkeypatch):
    mock = AsyncMock(return_value="こんにちは")
    monkeypatch.setattr("hermes_call.voice.run_process", mock)
    voice = Voice(config, None)
    assert asyncio.run(voice.reply("hermes-phone-1001", "1001", "後で電話して")) == "こんにちは"
    argv = mock.call_args.args[0]
    assert argv[argv.index("--continue") + 1] == "hermes-phone-1001"
    assert argv[argv.index("--toolsets") + 1] == "hermes_call,cronjob"
    assert "--create-if-missing" in argv
    assert "--quiet" in argv
    assert argv[argv.index("--format") + 1] == "stream-json"
    assert mock.call_args.kwargs["stream_json"] is True


def test_linked_chat_resumes_original_history_without_overriding_frozen_tools(config, monkeypatch):
    mock = AsyncMock(return_value="前のチャットの続きですね")
    monkeypatch.setattr("hermes_call.voice.run_process", mock)
    voice = Voice(config, None)
    asyncio.run(voice.reply("chat-original-123", "1001", "さっきの続き", linked=True))
    argv = mock.call_args.args[0]
    assert argv[argv.index("--resume") + 1] == "chat-original-123"
    assert "--create-if-missing" not in argv
    assert "--toolsets" not in argv


def test_subprocess_errors_and_timeout():
    assert asyncio.run(run_process([sys.executable, "-c", "print('ok')"], 5)) == "ok"
    with pytest.raises(RuntimeError, match="exit 3"):
        asyncio.run(run_process([sys.executable, "-c", "raise SystemExit(3)"], 5))
    with pytest.raises(TimeoutError):
        asyncio.run(run_process([sys.executable, "-c", "import time; time.sleep(30)"], 1))


def test_fragmented_http_json_is_read_to_eof():
    async def scenario():
        first_sent, continue_response = asyncio.Event(), asyncio.Event()

        async def stream(request):
            response = web.StreamResponse(headers={"Content-Type": "application/json"})
            await response.prepare(request)
            await response.write(b'{"outputSamplingRate":24000,"text":"')
            first_sent.set()
            await continue_response.wait()
            await response.write("こんにちは".encode())
            await response.write(b'"}')
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_get("/query", stream)
        async with TestServer(app) as server, aiohttp.ClientSession() as session:
            async with session.get(server.make_url("/query")) as response:
                await first_sent.wait()
                task = asyncio.create_task(read_json_object(response, 65536, "test audio_query"))
                await asyncio.sleep(0.01)
                assert not task.done()  # The first network fragment is not a whole JSON document.
                continue_response.set()
                result = await asyncio.wait_for(task, 1)
                assert result == {"outputSamplingRate": 24000, "text": "こんにちは"}

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "body,limit,expected",
    [
        (b'{"secret":"private transcript",', 1000, "invalid JSON"),
        (b'{"a":' + b" " * 100, 10, "exceeded"),
        (b"[]", 1000, "expected a JSON object"),
    ],
)
def test_invalid_provider_json_is_bounded_and_private(body, limit, expected, caplog):
    with pytest.raises(SpeechResponseError, match=expected) as caught:
        asyncio.run(read_json_object(SpeechResponse(body), limit, "VOICEVOX audio_query"))
    assert "private transcript" not in str(caught.value)
    assert "private transcript" not in caplog.text
