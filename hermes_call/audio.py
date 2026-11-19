"""Bounded PCM exchange between the native media clock and asyncio conversation loop."""

import asyncio
import struct
import threading
import time
from collections import deque

RATE = 16000
BYTES_PER_SECOND = RATE * 2


class AudioBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.incoming: deque[bytes] = deque()
        self.incoming_size = 0
        self.capture = False
        self.overflow = False
        self.outgoing = b""
        self.position = 0
        self.effect_intro = b""
        self.effect_loop = b""
        self.effect_position = 0
        self.effect_loop_position = 0
        self.ended = threading.Event()

    def receive(self, pcm: bytes) -> None:
        with self.lock:
            if self.capture:
                if self.incoming_size + len(pcm) > BYTES_PER_SECOND * 2:
                    self.overflow = True
                    return
                self.incoming.append(pcm)
                self.incoming_size += len(pcm)

    def provide(self, size: int) -> bytes:
        with self.lock:
            if self.ended.is_set():
                return b"\0" * size
            if self.position < len(self.outgoing):
                chunk = self.outgoing[self.position : self.position + size]
                self.position += len(chunk)
                return chunk.ljust(size, b"\0")
            chunk = self.effect_intro[self.effect_position : self.effect_position + size]
            self.effect_position += len(chunk)
            while len(chunk) < size and self.effect_loop:
                count = min(size - len(chunk), len(self.effect_loop) - self.effect_loop_position)
                chunk += self.effect_loop[
                    self.effect_loop_position : self.effect_loop_position + count
                ]
                self.effect_loop_position = (self.effect_loop_position + count) % len(
                    self.effect_loop
                )
            return chunk.ljust(size, b"\0")

    def set_effect(self, intro: bytes, loop: bytes = b"") -> None:
        if any(len(value) % 2 or len(value) > BYTES_PER_SECOND * 4 for value in (intro, loop)):
            raise ValueError("effects must be bounded PCM16")
        with self.lock:
            if self.ended.is_set():
                return
            self.effect_intro, self.effect_loop = intro, loop
            self.effect_position = self.effect_loop_position = 0

    def stop(self) -> None:
        self.ended.set()
        with self.lock:
            self.capture = False
            self.outgoing = b""
            self.position = 0
            self.effect_intro = self.effect_loop = b""

    async def play(self, pcm: bytes) -> None:
        if len(pcm) > BYTES_PER_SECOND * 180 or len(pcm) % 2:
            raise ValueError("playback must be bounded PCM16")
        with self.lock:
            self.capture = False
            self.effect_intro = self.effect_loop = b""
            self.outgoing, self.position = pcm, 0
        try:
            async with asyncio.timeout(185):
                while not self.ended.is_set():
                    with self.lock:
                        finished = self.position >= len(self.outgoing)
                    if finished:
                        # Allow the final native frame to reach the RTP sender.
                        await asyncio.sleep(0.06)
                        return
                    await asyncio.sleep(0.02)
                raise ConnectionError("call disconnected")
        finally:
            with self.lock:
                self.outgoing, self.position = b"", 0

    async def listen(
        self, limit_seconds: float = 30, silence_seconds: float = 2, wait_seconds: float = 10
    ) -> bytes:
        with self.lock:
            self.incoming.clear()
            self.incoming_size = 0
            self.overflow = False
            self.capture = True
        audio = bytearray()
        preroll = bytearray()
        max_bytes = int(BYTES_PER_SECOND * limit_seconds) & ~1
        preroll_limit = min(BYTES_PER_SECOND * 3 // 10, max_bytes) & ~1
        silence = 0
        started = time.monotonic()
        speech_started: float | None = None
        voiced = False
        try:
            while (
                (time.monotonic() - speech_started < limit_seconds)
                if speech_started is not None
                else (time.monotonic() - started < min(wait_seconds, limit_seconds))
            ):
                if self.ended.is_set():
                    raise ConnectionError("call disconnected")
                with self.lock:
                    if self.overflow:
                        raise RuntimeError("audio consumer fell behind")
                    chunks = list(self.incoming)
                    self.incoming.clear()
                    self.incoming_size = 0
                for chunk in chunks:
                    if len(chunk) % 2:
                        raise RuntimeError("invalid PCM frame")
                    energy = sum(x[0] ** 2 for x in struct.iter_unpack("<h", chunk))
                    is_voiced = bool(chunk) and energy / (len(chunk) // 2) >= 200**2
                    if not voiced:
                        if not is_voiced:
                            preroll.extend(chunk)
                            if len(preroll) > preroll_limit:
                                del preroll[: len(preroll) - preroll_limit]
                            continue
                        voiced, silence = True, 0
                        speech_started = time.monotonic()
                        keep = min(len(preroll), max(0, (max_bytes - len(chunk)) & ~1))
                        if keep:
                            audio.extend(preroll[-keep:])
                        preroll.clear()
                    if len(audio) + len(chunk) > max_bytes:
                        return bytes(audio)
                    audio.extend(chunk)
                    if is_voiced:
                        silence = 0
                    else:
                        silence += len(chunk)
                    if silence >= BYTES_PER_SECOND * silence_seconds:
                        return bytes(audio)
                if not voiced and time.monotonic() - started >= wait_seconds:
                    return b""
                await asyncio.sleep(0.02)
            return bytes(audio) if voiced else b""
        finally:
            with self.lock:
                self.capture = False
                self.incoming.clear()
                self.incoming_size = 0
