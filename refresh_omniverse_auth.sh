#!/usr/bin/env bash
# Refresh Omniverse/Nucleus auth for IsaacLab headless jobs.
#
# Usage:
#   ./refresh_omniverse_auth.sh [OPTIONS] [OMNIVERSE_URL ...]
#
# Options:
#   --server URL           Nucleus server URL to authenticate against
#                          (default: omniverse://isaac-dev.ov.nvidia.com)
#   --isaac-sim-root DIR   Isaac Sim standalone root
#                          (default: /localhome/local-pengfeig/pengfeig/isaac-sim-standalone-6.0.0-rc.22)
#   --timeout SEC          Max seconds to wait for auth/verification (default: 900)
#   --interval SEC         Poll interval in seconds after failed verification (default: 5)
#   --sign-out             Sign out first, forcing a fresh device-code flow
#   --no-default-urls      Do not verify the built-in factory/ORCA scene URLs
#   -h, --help             Show this help
#
# Default behavior:
#   Registers omni.client's device-flow auth callback, prints the login URL and
#   device code when auth is required, then verifies these common scene assets:
#     - factory.usd
#     - main_new_light.usd
#
# Example:
#   ./refresh_omniverse_auth.sh
#
# Example forcing a fresh code:
#   ./refresh_omniverse_auth.sh --sign-out
#
# Example checking a specific scene:
#   ./refresh_omniverse_auth.sh \
#     omniverse://isaac-dev.ov.nvidia.com/Library/IsaacHealthcare/0.5.0/Props/OrcaScenes/Scene1MX2/rlinf_scenes/factory.usd

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SERVER="omniverse://isaac-dev.ov.nvidia.com"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-/localhome/local-pengfeig/pengfeig/isaac-sim-standalone-6.0.0-rc.22}"
TIMEOUT_SEC=900
INTERVAL_SEC=5
SIGN_OUT=0
USE_DEFAULT_URLS=1
VERIFY_URLS=()

DEFAULT_URLS=(
  "omniverse://isaac-dev.ov.nvidia.com/Library/IsaacHealthcare/0.5.0/Props/OrcaScenes/Scene1MX2/rlinf_scenes/factory.usd"
  "omniverse://isaac-dev.ov.nvidia.com/Library/IsaacHealthcare/0.5.0/Props/OrcaScenes/Scene1MX2/main_new_light.usd"
)

usage() {
  sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server)
      SERVER="$2"
      shift 2
      ;;
    --server=*)
      SERVER="${1#*=}"
      shift
      ;;
    --isaac-sim-root)
      ISAAC_SIM_ROOT="$2"
      shift 2
      ;;
    --isaac-sim-root=*)
      ISAAC_SIM_ROOT="${1#*=}"
      shift
      ;;
    --timeout)
      TIMEOUT_SEC="$2"
      shift 2
      ;;
    --timeout=*)
      TIMEOUT_SEC="${1#*=}"
      shift
      ;;
    --interval)
      INTERVAL_SEC="$2"
      shift 2
      ;;
    --interval=*)
      INTERVAL_SEC="${1#*=}"
      shift
      ;;
    --sign-out)
      SIGN_OUT=1
      shift
      ;;
    --no-default-urls)
      USE_DEFAULT_URLS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        VERIFY_URLS+=("$1")
        shift
      done
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      VERIFY_URLS+=("$1")
      shift
      ;;
  esac
done

if [[ "$USE_DEFAULT_URLS" == "1" ]]; then
  VERIFY_URLS=("${DEFAULT_URLS[@]}" "${VERIFY_URLS[@]}")
fi

if [[ ${#VERIFY_URLS[@]} -eq 0 ]]; then
  echo "No verification URLs provided. Add OMNIVERSE_URL args or omit --no-default-urls." >&2
  exit 2
fi

KIT_PYTHON="${ISAAC_SIM_ROOT}/kit/python.sh"
OMNI_CLIENT_LIB="${ISAAC_SIM_ROOT}/kit/extscore/omni.client.lib"

if [[ ! -x "$KIT_PYTHON" ]]; then
  echo "Isaac Sim kit python not found or not executable: $KIT_PYTHON" >&2
  exit 1
fi

if [[ ! -d "$OMNI_CLIENT_LIB" ]]; then
  echo "omni.client Python library not found: $OMNI_CLIENT_LIB" >&2
  exit 1
fi

export PYTHONPATH="${OMNI_CLIENT_LIB}${PYTHONPATH:+:${PYTHONPATH}}"

cd "$SCRIPT_DIR"

"$KIT_PYTHON" - "$SERVER" "$TIMEOUT_SEC" "$INTERVAL_SEC" "$SIGN_OUT" "${VERIFY_URLS[@]}" <<'PYCODE'
import sys
import time

import omni.client


def _is_ok(result):
    return result == omni.client.Result.OK or str(result).endswith("OK")


server = sys.argv[1]
timeout_sec = int(float(sys.argv[2]))
interval_sec = float(sys.argv[3])
sign_out = sys.argv[4] == "1"
urls = sys.argv[5:]


def on_device_auth(auth_handle, params):
    if params is None:
        print(f"DEVICE_AUTH_FINISHED handle={auth_handle}", flush=True)
        return

    print("\n=== OMNIVERSE_DEVICE_AUTH_REQUIRED ===", flush=True)
    print(f"AUTH_SERVER={params.server}", flush=True)
    print(f"AUTH_URL={params.url}", flush=True)
    print(f"AUTH_CODE={params.code}", flush=True)
    print(f"AUTH_EXPIRES_IN_SEC={params.expiration}", flush=True)
    print("Open AUTH_URL in a browser and enter AUTH_CODE.", flush=True)
    print("=== END_DEVICE_AUTH ===\n", flush=True)



def on_connection_status(server_name, status):
    print(f"CONNECTION_STATUS server={server_name} status={status}", flush=True)


device_auth_reg = omni.client.register_device_flow_auth_callback(on_device_auth)
status_reg = omni.client.register_connection_status_callback(on_connection_status)
omni.client.initialize()

if sign_out:
    print(f"SIGN_OUT {server}", flush=True)
    omni.client.sign_out(server)

print(f"SERVER={server}", flush=True)
print("VERIFY_URLS=", flush=True)
for url in urls:
    print(f"  {url}", flush=True)

# This helps recover from stale ERROR_CONNECTION after a previous failed browser auth attempt.
omni.client.reconnect(server)

deadline = time.time() + timeout_sec
attempt = 0
last_results = {}

while time.time() <= deadline:
    attempt += 1
    all_ok = True
    for url in urls:
        result, entry = omni.client.stat(url)
        last_results[url] = result
        print(f"ATTEMPT={attempt} RESULT={result} URL={url}", flush=True)
        if not _is_ok(result):
            all_ok = False

    if all_ok:
        result, info = omni.client.get_server_info(server)
        print(f"OMNI_AUTH_OK server_info={result} user={getattr(info, 'username', '')}", flush=True)
        sys.exit(0)

    time.sleep(interval_sec)

print("OMNI_AUTH_TIMEOUT", flush=True)
for url, result in last_results.items():
    print(f"FINAL_RESULT={result} URL={url}", flush=True)
sys.exit(2)
PYCODE
