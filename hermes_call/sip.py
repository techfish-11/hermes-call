"""Direct PJSUA2 user agent. All SIP objects live on the asyncio/main thread.

Only PCM copies run on the native audio clock thread. No HTTP, files or agent
operations execute inside native callbacks.
"""

import asyncio
import importlib
import ipaddress
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .audio import AudioBuffer
from .config import Config
from .store import new_id

log = logging.getLogger(__name__)


_ADDR_SPEC = re.compile(
    r"sips?:([0-9]{1,12})@([A-Za-z0-9.-]+)(?::[0-9]+)?(?:;[^\s>]*)?\Z",
    re.IGNORECASE,
)


def _extract_name_addr(text: str) -> str | None:
    quoted = escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
        elif char == "\\" and quoted:
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif not quoted and char == "<":
            end = text.find(">", index + 1)
            if end < 0:
                return None
            rest = text[end + 1 :].lstrip()
            if rest and not rest.startswith(";"):
                return None
            return text[index + 1 : end].strip()
    return None


def caller_identity(uri: str) -> tuple[str | None, str | None]:
    text = uri.strip()
    if (addr := _extract_name_addr(text)) is None:
        if "<" in text or ">" in text:
            return (None, None)
        addr = text
    match = _ADDR_SPEC.fullmatch(addr)
    return (match[1], match[2].lower()) if match else (None, None)


def caller_extension(uri: str, domain: str) -> str | None:
    extension, actual_domain = caller_identity(uri)
    if actual_domain == domain.lower():
        return extension
    return None


def peer_address(source: str) -> str | None:
    try:
        host, separator, port = source.rpartition(":")
        if not separator or not port.isdigit():
            return None
        return str(ipaddress.IPv4Address(host))
    except ipaddress.AddressValueError:
        return None


@dataclass(frozen=True)
class IncomingDecision:
    code: int
    reason: str
    extension: str | None
    domain: str | None
    peer: str | None


def incoming_decision(
    config: Config, uri: str, source: str, *, closing: bool, busy: bool, audio_ready: bool
) -> IncomingDecision:
    extension, domain = caller_identity(uri)
    peer = peer_address(source)
    domains = config.allowed_caller_domains or (config.sip_domain,)
    if closing:
        code, reason = 503, "service_stopping"
    elif peer is None:
        code, reason = 403, "invalid_peer_address"
    elif peer not in config.sip_peer_ips:
        code, reason = 403, "peer_not_allowed"
    elif extension is None:
        code, reason = 403, "invalid_caller_uri"
    elif domain not in {value.lower() for value in domains}:
        code, reason = 403, "caller_domain_not_allowed"
    elif extension not in config.allowed_callers:
        code, reason = 403, "extension_not_allowed"
    elif busy:
        code, reason = 486, "line_busy"
    elif not audio_ready:
        code, reason = 503, "initial_audio_not_ready"
    else:
        code, reason = 200, "accepted"
    return IncomingDecision(code, reason, extension, domain, peer)


def native_error(exc: Exception) -> str:
    """Identify the failing native API/attribute without dumping SIP data or credentials."""
    parts = [type(exc).__name__]
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        parts.append(f"status={status}")
    for name in ("title", "name"):
        value = getattr(exc, name, None)
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_():., -]{1,160}", value):
            parts.append(f"{name}={value}")
    return " ".join(parts)


def media_snapshot(info) -> tuple[tuple[int, int, int], ...]:
    """Keep SWIG's owning CallInfo alive while copying its borrowed media vector."""
    return tuple((int(item.index), int(item.type), int(item.status)) for item in info.media)


class MediaLink:
    """Own one idempotent bidirectional conference link; cache slot IDs by value."""

    # PJ_EINVAL and PJ_ENOTFOUND: an old conference slot/link was already removed by PJSIP.
    STALE_SLOT_STATUSES = frozenset((70004, 70006))

    def __init__(self, port, native_error_type):
        self.port = port
        self.native_error_type = native_error_type
        self.media = None
        self.slot = None

    def disconnect(self) -> None:
        if self.media is None:
            return
        for source, sink in ((self.media, self.port), (self.port, self.media)):
            try:
                source.stopTransmit(sink)
            except self.native_error_type as exc:
                if getattr(exc, "status", None) not in self.STALE_SLOT_STATUSES:
                    raise
                log.info("old audio connection already absent (%s)", native_error(exc))
        self.media, self.slot = None, None

    def connect(self, media) -> bool:
        slot = media.getPortId()
        if slot < 0:
            raise RuntimeError("audio conference slot is not ready")
        if self.slot == slot:
            return False
        self.disconnect()
        media.startTransmit(self.port)
        try:
            self.port.startTransmit(media)
        except Exception:
            # Roll back the receive connection if playback could not be connected.
            try:
                media.stopTransmit(self.port)
            except self.native_error_type as rollback_error:
                log.error("audio connect rollback failed (%s)", native_error(rollback_error))
            raise
        self.media, self.slot = media, slot
        return True


@dataclass
class CallSession:
    id: str
    extension: str
    message: str
    outbound: bool
    audio: AudioBuffer = field(default_factory=AudioBuffer)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    answered: asyncio.Event = field(default_factory=asyncio.Event)
    disconnected: asyncio.Event = field(default_factory=asyncio.Event)
    native: Any = None
    status: int = 0
    confirmed: bool = False
    native_disconnected: bool = False
    confirmed_at: float | None = None
    initial_audio: bytes | None = None
    chat_ref: str | None = None
    failure_reason: str | None = None
    media_connected: bool = False
    release_on_disconnect: bool = False


def _abort_session(session: CallSession | None, reason: str) -> None:
    """Stop audio and unblock all waiters when a native SIP callback fails."""
    if session is None:
        return
    session.failure_reason = reason
    session.audio.stop()
    session.disconnected.set()
    session.ready.set()
    session.answered.set()


class SipClient:
    def __init__(self, config: Config, on_call: Callable[[CallSession], None]):
        self.config, self.on_call = config, on_call
        self.registered = False
        self.registered_event = asyncio.Event()
        self.registration_status = 0
        self.sessions: dict[str, CallSession] = {}
        self.native_calls: dict[int, Any] = {}
        self.ep = None
        self.account = None
        self.pj = None
        self.closing = False
        self.reserved: str | None = None
        self.inbound_audio: bytes | None = None

    @property
    def busy(self) -> bool:
        return bool(self.sessions) or self.reserved is not None

    def start(self) -> None:
        pj = self.pj = importlib.import_module("pjsua2")
        owner = self
        loop = asyncio.get_running_loop()

        class Port(pj.AudioMediaPort):
            def __init__(self, audio):
                super().__init__()
                self.audio = audio

            def onFrameReceived(self, frame):
                if frame.type == pj.PJMEDIA_FRAME_TYPE_AUDIO:
                    self.audio.receive(bytes(frame.buf))

            def onFrameRequested(self, frame):
                frame.type = pj.PJMEDIA_FRAME_TYPE_AUDIO
                frame.buf.assign_from_bytes(self.audio.provide(frame.size))

        class Call(pj.Call):
            def __init__(self, account, call_id=pj.PJSUA_INVALID_ID, session=None):
                super().__init__(account, call_id)
                self.session = session
                self.port = None
                self.media = None
                self.media_index = None
                self.link = None

            def onCallState(self, prm):
                try:
                    info = self.getInfo()
                    session = self.session
                    log.info(
                        "SIP call native_id=%s state=%s status=%s",
                        self.getId(),
                        info.state,
                        info.lastStatusCode,
                    )
                    if session:
                        session.status = info.lastStatusCode
                        if info.state == pj.PJSIP_INV_STATE_CONFIRMED:
                            if session.confirmed_at is None:
                                session.confirmed_at = loop.time()
                            session.confirmed = True
                            session.answered.set()
                            # Media may already be active before our incoming Call wrapper exists.
                            self.connect_media()
                    if info.state == pj.PJSIP_INV_STATE_DISCONNECTED:
                        if session:
                            session.native_disconnected = True
                            session.audio.stop()
                            session.disconnected.set()
                            session.ready.set()  # Wake an unanswered/media wait immediately.
                            session.answered.set()
                            if session.release_on_disconnect:
                                owner.release(session)
                        # Keep SWIG objects alive until the native callback returns.
                        loop.call_soon(owner.native_calls.pop, self.getId(), None)
                except Exception as exc:
                    log.error("SIP state callback failed (%s)", native_error(exc))
                    _abort_session(self.session, "state_callback_error")

            def onCallMediaState(self, prm):
                self.connect_media()

            def connect_media(self):
                try:
                    if not self.session or self.session.disconnected.is_set():
                        return
                    # Do not iterate self.getInfo().media: the temporary owner can be destroyed
                    # while SWIG's vector/element wrappers still refer to its C++ memory.
                    snapshot = media_snapshot(self.getInfo())
                    log.info("call %s media snapshot=%s", self.session.id, snapshot)
                    for index, kind, status in snapshot:
                        if kind == pj.PJMEDIA_TYPE_AUDIO and status == pj.PJSUA_CALL_MEDIA_ACTIVE:
                            media = self.getAudioMedia(index)
                            if self.port is None:
                                self.port = Port(self.session.audio)
                                fmt = pj.MediaFormatAudio()
                                fmt.type, fmt.clockRate = pj.PJMEDIA_TYPE_AUDIO, 16000
                                fmt.channelCount, fmt.bitsPerSample = 1, 16
                                fmt.frameTimeUsec = 20000
                                self.port.createPort("hermes-" + self.session.id, fmt)
                                self.link = MediaLink(self.port, pj.Error)
                            changed = self.link.connect(media)
                            self.media = media
                            self.media_index = index
                            self.session.ready.set()
                            self.session.media_connected = True
                            if changed:
                                log.info("call %s audio media ready", self.session.id)
                            break
                    else:
                        if self.link is not None:
                            self.link.disconnect()
                        self.session.media_connected = False
                except Exception as exc:
                    log.error("SIP media callback failed (%s)", native_error(exc))
                    _abort_session(self.session, "media_callback_error")

        class Account(pj.Account):
            def onRegState(self, prm):
                info = self.getInfo()
                owner.registered = info.regIsActive and 200 <= info.regStatus < 300
                owner.registration_status = info.regStatus
                if owner.registered:
                    owner.registered_event.set()
                else:
                    owner.registered_event.clear()
                log.info("SIP registration status=%s active=%s", info.regStatus, owner.registered)

            def onIncomingCall(self, prm):
                call = Call(self, prm.callId)
                owner.native_calls[prm.callId] = call
                try:
                    decision = incoming_decision(
                        owner.config,
                        call.getInfo().remoteUri,
                        prm.rdata.srcAddress,
                        closing=owner.closing,
                        busy=owner.busy,
                        audio_ready=bool(owner.inbound_audio),
                    )
                    code, extension = decision.code, decision.extension
                    log.log(
                        logging.INFO if code == 200 else logging.WARNING,
                        "SIP incoming native_id=%s peer=%s extension=%s caller_domain=%s "
                        "response=%s reason=%s",
                        prm.callId,
                        decision.peer,
                        extension,
                        decision.domain,
                        code,
                        decision.reason,
                    )
                    if code == 200:
                        session = CallSession(new_id(), extension, "", False)
                        session.native = call
                        session.initial_audio = owner.inbound_audio
                        call.session = session
                        owner.sessions[session.id] = session
                        owner.on_call(session)
                    op = pj.CallOpParam()
                    op.statusCode = code
                    op.opt.audioCount, op.opt.videoCount = 1, 0
                    call.answer(op)
                except Exception as exc:
                    log.error("SIP incoming callback failed (%s)", native_error(exc))
                    _abort_session(call.session, "incoming_callback_error")
                    op = pj.CallOpParam()
                    op.statusCode = 500
                    try:
                        call.hangup(op)
                    except Exception as hangup_error:
                        log.error("SIP reject/hangup failed (%s)", native_error(hangup_error))

        self.call_class = Call
        self.ep = pj.Endpoint()
        self.ep.libCreate()
        cfg = pj.EpConfig()
        cfg.uaConfig.threadCnt = 0
        cfg.uaConfig.mainThreadOnly = True
        cfg.uaConfig.maxCalls = 4  # Room to reject extra incoming dialogs cleanly.
        cfg.uaConfig.userAgent = "HermesCall/0.2"
        for server in self.config.stun_servers:
            cfg.uaConfig.stunServer.append(server)
        cfg.medConfig.clockRate, cfg.medConfig.sndClockRate = 16000, 16000
        cfg.medConfig.channelCount = 1
        cfg.medConfig.sndAutoCloseTime = -1  # Keep the headless conference clock running.
        cfg.logConfig.level, cfg.logConfig.consoleLevel = 0, 0
        self.ep.libInit(cfg)
        transport = pj.TransportConfig()
        transport.port = self.config.sip_local_port
        transport.boundAddress = self.config.sip_bind_address
        transport.tlsConfig.verifyServer = True
        transport.tlsConfig.CaListFile = "/etc/ssl/certs/ca-certificates.crt"
        kind = {
            "udp": pj.PJSIP_TRANSPORT_UDP,
            "tcp": pj.PJSIP_TRANSPORT_TCP,
            "tls": pj.PJSIP_TRANSPORT_TLS,
        }[self.config.sip_transport]
        transport_id = self.ep.transportCreate(kind, transport)
        self.ep.libStart()
        self.ep.audDevManager().setNullDev()  # Supplies a clock without ALSA or sound hardware.
        # Prefer widely interoperable G.711 audio; disable everything else.
        for codec in self.ep.codecEnum2():
            self.ep.codecSetPriority(
                codec.codecId, 200 if codec.codecId in ("PCMU/8000/1", "PCMA/8000/1") else 0
            )
        c = self.config
        account_cfg = pj.AccountConfig()
        scheme = "sips" if c.sip_transport == "tls" else "sip"
        account_cfg.idUri = f"{scheme}:{c.sip_username}@{c.sip_domain}"
        server_uri = f"{scheme}:{c.sip_server}:{c.sip_port};transport={c.sip_transport}"
        account_cfg.regConfig.registrarUri = server_uri
        account_cfg.regConfig.timeoutSec = 300
        account_cfg.regConfig.retryIntervalSec = 30
        account_cfg.sipConfig.transportId = transport_id
        account_cfg.sipConfig.proxies.append(server_uri + ";lr")
        account_cfg.sipConfig.authCreds.append(
            pj.AuthCredInfo("digest", "*", c.sip_username, 0, c.sip_password)
        )
        account_cfg.mediaConfig.transportConfig.port = c.rtp_port
        account_cfg.mediaConfig.transportConfig.portRange = 20
        account_cfg.mediaConfig.srtpUse = (
            pj.PJMEDIA_SRTP_MANDATORY if c.sip_srtp_required else pj.PJMEDIA_SRTP_DISABLED
        )
        self.account = Account()
        self.account.create(account_cfg)

    async def poll(self) -> None:
        while not self.closing:
            self.ep.libHandleEvents(0)
            await asyncio.sleep(0.01)

    def dial(
        self,
        call_id: str,
        extension: str,
        message: str,
        initial_audio: bytes,
        chat_ref: str | None = None,
    ) -> CallSession:
        if not self.registered:
            raise ConnectionError("SIP account is not registered")
        if self.sessions or (self.reserved is not None and self.reserved != call_id):
            raise RuntimeError("telephone is busy")
        if extension not in self.config.allowed_extensions:
            raise ValueError("extension is not allowed")
        if not initial_audio or len(initial_audio) % 2:
            raise ValueError("initial message audio is mandatory")
        session = CallSession(call_id, extension, message, True)
        session.initial_audio, session.chat_ref = initial_audio, chat_ref
        call = self.call_class(self.account, session=session)
        session.native = call
        self.sessions[call_id] = session
        on_call_attempted = False
        notified = False
        try:
            op = self.pj.CallOpParam(True)
            op.opt.audioCount, op.opt.videoCount = 1, 0
            scheme = "sips" if self.config.sip_transport == "tls" else "sip"
            call.makeCall(f"{scheme}:{extension}@{self.config.sip_domain}", op)
            native_id = call.getId()
            if native_id != self.pj.PJSUA_INVALID_ID:
                self.native_calls[native_id] = call
            on_call_attempted = True
            self.on_call(session)
            notified = True
            return session
        except Exception:
            native_id = call.getId()
            if native_id == self.pj.PJSUA_INVALID_ID:
                self.release(session)
                raise
            self.native_calls[native_id] = call
            if not on_call_attempted:
                # Keep ownership and the busy slot until bounded conversation cleanup runs.
                try:
                    self.on_call(session)
                    notified = True
                except Exception as callback_error:
                    log.error("SIP failed dial handoff failed (%s)", native_error(callback_error))
            if not notified:
                # Handoff failed: attempt hangup, and keep the session/busy slot
                # until native disconnect is confirmed.
                session.audio.stop()
                session.release_on_disconnect = True
                try:
                    self.hangup(session)
                except Exception as hangup_error:
                    log.warning("SIP failed dial hangup failed (%s)", native_error(hangup_error))
                if session.native_disconnected:
                    self.release(session)
            raise

    def hangup(self, session: CallSession) -> None:
        if not session.native_disconnected and session.native is not None:
            op = self.pj.CallOpParam()
            session.native.hangup(op)

    def release(self, session: CallSession) -> None:
        session.audio.stop()
        self.sessions.pop(session.id, None)
        if session.native:
            session.native.session = None
        session.native = None

    async def close(self) -> None:
        self.closing = True
        if self.ep is None:
            return
        for session in list(self.sessions.values()):
            try:
                self.hangup(session)
            except Exception as exc:
                log.warning("SIP hangup failed (%s)", type(exc).__name__)
            session.audio.stop()
        try:
            if self.account:
                try:
                    self.account.setRegistration(False)
                    for _ in range(50):
                        self.ep.libHandleEvents(0)
                        await asyncio.sleep(0.01)
                finally:
                    self.account.shutdown()
        finally:
            # Destroy wrappers on the owning thread, before destroying the native endpoint.
            for session in list(self.sessions.values()):
                self.release(session)
            self.native_calls.clear()
            self.account = None
            self.ep.libDestroy()
            self.ep = None
