"""Owner-only configuration for a persistent, directly registered SIP user agent."""

import ipaddress
import json
import math
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

EXTENSION = re.compile(r"[0-9]{1,12}\Z")


@dataclass(frozen=True)
class Config:
    api_token: str = field(repr=False)
    sip_password: str = field(repr=False)
    stt_api_key: str = field(repr=False)
    sip_server: str
    sip_domain: str
    sip_username: str
    sip_peer_ips: tuple[str, ...]
    allowed_extensions: tuple[str, ...]
    allowed_callers: tuple[str, ...]
    hermes_command: tuple[str, ...]
    data_dir: Path = Path("/var/lib/hermes-call")
    api_port: int = 8765
    sip_port: int = 5060
    sip_local_port: int = 5068
    sip_transport: str = "udp"
    sip_bind_address: str = "0.0.0.0"
    sip_srtp_required: bool = False
    stun_servers: tuple[str, ...] = ()
    rtp_port: int = 40000
    stt_url: str = "https://api.groq.com/openai/v1"
    stt_model: str = "whisper-large-v3-turbo"
    tts_url: str = "https://voicevox-api.swiftlybot.com/v1"
    tts_speaker: int = 3
    tts_max_text_chars: int = 500
    language: str = "ja"
    max_call_seconds: int = 1800
    agent_timeout_seconds: int = 120
    hermes_toolsets: tuple[str, ...] = ("hermes_call", "cronjob")
    incoming_message: str = "Hermesです。前の会話の続きもお話しできます。ご用件をどうぞ。"
    allowed_caller_domains: tuple[str, ...] = ()
    sound_effects: bool = True
    sound_volume: float = 0.10

    def __post_init__(self) -> None:
        if type(self.sound_effects) is not bool:
            raise ValueError("sound_effects must be boolean")
        if (
            type(self.sound_volume) not in (int, float)
            or not math.isfinite(self.sound_volume)
            or not 0 <= self.sound_volume <= 0.3
        ):
            raise ValueError("sound_volume must be a finite number in 0..0.3")
        if (
            not isinstance(self.incoming_message, str)
            or not self.incoming_message.strip()
            or len(self.incoming_message) > 2000
        ):
            raise ValueError("incoming_message must contain 1..2000 characters")
        for name in ("api_token", "sip_password", "stt_api_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or value.startswith("REPLACE_"):
                raise ValueError(f"{name} must be configured")
        if len(self.api_token) < 32:
            raise ValueError("api_token must contain at least 32 characters")
        for name in ("sip_server", "sip_domain"):
            value = getattr(self, name)
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9.-]{1,253}", value):
                raise ValueError(f"{name} must be a hostname or IPv4 address, without a port")
        for domain in self.allowed_caller_domains:
            if not isinstance(domain, str) or not re.fullmatch(r"[A-Za-z0-9.-]{1,253}", domain):
                raise ValueError("allowed_caller_domains must contain hostnames or IPv4 addresses")
        if not isinstance(self.sip_username, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]{1,100}", self.sip_username
        ):
            raise ValueError("invalid sip_username")
        for name in ("allowed_extensions", "allowed_callers"):
            values = getattr(self, name)
            if not values or any(
                not isinstance(v, str) or not EXTENSION.fullmatch(v) for v in values
            ):
                raise ValueError(f"{name} must contain numeric extensions")
        if not self.sip_peer_ips:
            raise ValueError("sip_peer_ips must contain trusted PBX addresses")
        for address in (*self.sip_peer_ips, self.sip_bind_address):
            if not isinstance(address, str):
                raise ValueError("SIP IPv4 addresses must be strings")
            ipaddress.IPv4Address(address)
        if self.sip_transport not in ("udp", "tcp", "tls"):
            raise ValueError("sip_transport must be udp, tcp or tls")
        if type(self.sip_srtp_required) is not bool:
            raise ValueError("sip_srtp_required must be boolean")
        if self.sip_srtp_required and self.sip_transport != "tls":
            raise ValueError("mandatory SRTP requires TLS signaling")
        if not self.hermes_command or any(
            not isinstance(v, str) or not v for v in self.hermes_command
        ):
            raise ValueError("hermes_command must be an argv list")
        if not Path(self.hermes_command[0]).is_absolute():
            raise ValueError("hermes_command executable must be absolute")
        if not isinstance(self.data_dir, Path) or not self.data_dir.is_absolute():
            raise ValueError("data_dir must be absolute")
        for value in self.hermes_toolsets:
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
                raise ValueError("invalid hermes_toolsets")
        for value in self.stun_servers:
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9.:-]{1,253}", value):
                raise ValueError("invalid stun_servers")
        for name, low, high in (
            ("api_port", 1024, 65535),
            ("sip_port", 1, 65535),
            ("sip_local_port", 1024, 65535),
            ("rtp_port", 1024, 65500),
            ("tts_speaker", 0, 100000),
            ("tts_max_text_chars", 50, 1000),
            ("max_call_seconds", 60, 7200),
            ("agent_timeout_seconds", 10, 300),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"invalid {name}")
        if self.api_port == self.sip_local_port or self.rtp_port % 2:
            raise ValueError("API/SIP ports must differ; RTP base port must be even")
        for name in ("stt_url", "tts_url"):
            url = getattr(self, name)
            if not isinstance(url, str):
                raise ValueError(f"{name} must be a URL string")
            parsed = urlsplit(url)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
                or url.endswith("/")
            ):
                raise ValueError(f"{name} must be HTTPS without credentials/query/trailing slash")
        for name in ("stt_model", "language"):
            value = getattr(self, name)
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", value):
                raise ValueError(f"invalid {name}")

    @classmethod
    def load(cls, path: Path) -> "Config":
        if os.name == "posix":
            info = path.stat()
            if stat.S_IMODE(info.st_mode) & 0o077 or info.st_uid != os.getuid():
                raise ValueError("config must be owned by service user with mode 600")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("config must be a JSON object")
        required = ("sip_peer_ips", "allowed_extensions", "allowed_callers", "hermes_command")
        for name in (*required, "hermes_toolsets", "stun_servers", "allowed_caller_domains"):
            if name in raw or name in required:
                if not isinstance(raw.get(name), list):
                    raise ValueError(f"{name} must be a list")
                raw[name] = tuple(raw[name])
        if "data_dir" in raw:
            if not isinstance(raw["data_dir"], str):
                raise ValueError("data_dir must be a path string")
            raw["data_dir"] = Path(raw["data_dir"])
        try:
            return cls(**raw)
        except TypeError as exc:
            raise ValueError(
                "missing or obsolete configuration fields; see config.example.json"
            ) from exc
