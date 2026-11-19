import json
import os
import sys
from dataclasses import replace

import pytest

from hermes_call.config import Config


@pytest.mark.parametrize(
    "change",
    [
        {"sip_server": "pbx.example.com;bad=1"},
        {"stt_url": "http://api.groq.com/openai/v1"},
        {"tts_url": "http://example.com:50021"},
        {"api_token": "short"},
        {"sip_username": "pbx/123@evil"},
        {"allowed_callers": ()},
        {"allowed_extensions": ("123\n",)},
        {"sip_local_port": True},
        {"sip_peer_ips": ()},
        {"sip_peer_ips": ("not-an-IP",)},
        {"sip_srtp_required": True, "sip_transport": "udp"},
        {"rtp_port": 40001},
        {"stt_api_key": "REPLACE_WITH_GROQ_API_KEY"},
        {"hermes_command": ("hermes",)},
        {"incoming_message": " "},
        {"sound_effects": "true"},
        {"sound_volume": float("nan")},
        {"sound_volume": 0.31},
        {"sound_volume": True},
        {"sip_username": 7000},
        {"sip_peer_ips": (0,)},
        {"sip_bind_address": 0},
        {"tts_url": None},
        {"stt_url": 123},
        {"stt_model": 42},
        {"language": 123},
    ],
)
def test_invalid_config(config, change):
    with pytest.raises(ValueError):
        replace(config, **change)


def test_secrets_not_in_repr(config):
    assert config.stt_api_key not in repr(config)
    assert config.api_token not in repr(config)
    assert config.sip_password not in repr(config)


def _valid_raw_config(tmp_path):
    return {
        "api_token": "a" * 64,
        "sip_password": "sip-test-password-unique",
        "stt_api_key": "groq-test-key",
        "sip_server": "pbx.example.com",
        "sip_domain": "pbx.example.com",
        "sip_username": "7000",
        "sip_peer_ips": ["192.0.2.10"],
        "allowed_extensions": ["1001"],
        "allowed_callers": ["1001"],
        "hermes_command": [sys.executable],
        "data_dir": str(tmp_path),
    }


def test_config_load_reports_type_errors_without_misleading_field_message(tmp_path):
    path = tmp_path / "config.json"
    raw = _valid_raw_config(tmp_path)
    path.write_text(json.dumps(raw), encoding="utf-8")
    if os.name == "posix":
        path.chmod(0o644)
        with pytest.raises(ValueError, match="mode 600"):
            Config.load(path)
        path.chmod(0o600)

    loaded = Config.load(path)
    assert loaded.sip_username == "7000"

    for bad in ({"sip_username": 7000}, {"data_dir": 123}, {"tts_url": None}):
        path.write_text(json.dumps({**raw, **bad}), encoding="utf-8")
        with pytest.raises(ValueError) as exc_info:
            Config.load(path)
        assert "missing or obsolete" not in str(exc_info.value)

    path.write_text(json.dumps({**raw, "obsolete_key": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing or obsolete"):
        Config.load(path)

    missing = {k: v for k, v in raw.items() if k != "sip_server"}
    path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError, match="missing or obsolete"):
        Config.load(path)
