#!/usr/bin/env bash
# Run as the same non-root user that runs Hermes. No local PBX or sound hardware required.
set -euo pipefail
umask 077

if [[ $(id -u) -eq 0 ]]; then
  echo 'Run as your Hermes user, without sudo (the script uses sudo when needed).' >&2
  exit 1
fi
# shellcheck source=/dev/null
source /etc/os-release
if [[ ${ID:-} != ubuntu || ${VERSION_ID:-} != 24.04 ]]; then
  echo 'Supported installer target: Ubuntu Server 24.04.' >&2
  exit 1
fi
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
install_dir="$HOME/.local/share/hermes-call"
config_dir="$HOME/.config/hermes-call"
hermes_home="${HERMES_HOME:-$HOME/.hermes}"
# shellcheck source=scripts/find-hermes.sh
source "$repo_dir/scripts/find-hermes.sh"
hermes_bin=$(resolve_hermes)
printf 'Using Hermes launcher: %s\n' "$hermes_bin"
for local_path in "$HOME" "$install_dir" "$config_dir" "$hermes_home" "$hermes_bin"; do
  if [[ ! "$local_path" =~ ^/[a-zA-Z0-9_./-]+$ ]]; then
    echo 'Install paths must be absolute and contain only letters, digits, _, ., / and -.' >&2
    exit 1
  fi
done

if systemctl cat hermes-call.service >/dev/null 2>&1; then
  sudo systemctl stop hermes-call.service
fi

sudo apt-get update
sudo apt-get install -y ffmpeg python3-venv python3-dev build-essential swig \
  pkg-config libssl-dev libsrtp2-dev libasound2-dev git ca-certificates
python_bin=$(command -v python3)
"$python_bin" -c 'import sys; assert sys.version_info >= (3,11), "Python >=3.11 required"'
mkdir -p "$install_dir" "$config_dir" "$hermes_home/plugins/hermes-call"
"$python_bin" -m venv "$install_dir/venv"
"$install_dir/venv/bin/python" -m pip install 'setuptools>=75,<83' 'wheel>=0.45,<1'
bash "$repo_dir/scripts/build-pjsip.sh" "$install_dir/venv" "$install_dir/build"
"$install_dir/venv/bin/python" -m pip install --upgrade "$repo_dir"
"$install_dir/venv/bin/python" -m pip install --force-reinstall --no-deps "$repo_dir"
install -m 600 "$repo_dir/__init__.py" "$hermes_home/plugins/hermes-call/__init__.py"
install -m 600 "$repo_dir/plugin.yaml" "$hermes_home/plugins/hermes-call/plugin.yaml"

if [[ ! -f "$config_dir/config.json" ]]; then
  "$install_dir/venv/bin/python" - "$repo_dir/config.example.json" \
      "$config_dir/config.json" "$hermes_bin" <<'PY'
import json
import secrets
import sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())
config['api_token'] = secrets.token_hex(32)
config['hermes_command'] = [sys.argv[3]]
with Path(sys.argv[2]).open('x') as stream:
    json.dump(config, stream, indent=2)
    stream.write('\n')
PY
fi
chmod 600 "$config_dir/config.json"
sudo install -d -m 700 -o "$(id -un)" -g "$(id -gn)" /var/lib/hermes-call
sudo install -d -m 700 -o "$(id -un)" -g "$(id -gn)" /var/lib/hermes-call/media

unit_file=$(mktemp)
trap 'rm -f -- "$unit_file"' EXIT
cat > "$unit_file" <<EOF
[Unit]
Description=Hermes SIP telephone integration
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(id -un)
Group=$(id -gn)
Environment=HOME=$HOME
Environment=PYTHONUNBUFFERED=1
Environment=HERMES_HOME=$hermes_home
Environment=HERMES_CALL_CONFIG=$config_dir/config.json
Environment=PATH=${hermes_bin%/*}:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=$install_dir/venv/bin/hermes-call --config $config_dir/config.json
WorkingDirectory=$install_dir
Restart=on-failure
RestartSec=30
TimeoutStopSec=15
KillMode=control-group
UMask=0077
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/var/lib/hermes-call $hermes_home $config_dir $install_dir
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6

[Install]
WantedBy=multi-user.target
EOF
sudo install -m 644 "$unit_file" /etc/systemd/system/hermes-call.service
sudo systemctl daemon-reload
printf '\nInstalled. Next steps:\n'
printf '  1. Edit %s/config.json (SIP account, trusted PBX IPs, Groq key).\n' "$config_dir"
printf '  2. Allow SIP/RTP from your PBX in the firewall (see README).\n'
printf '  3. VOICEVOX uses https://voicevox-api.swiftlybot.com/v1; no local engine needed.\n'
printf '  4. hermes plugins enable hermes-call\n'
printf '  5. %s/venv/bin/hermes-call --check\n' "$install_dir"
printf '  6. sudo systemctl enable --now hermes-call\n'
