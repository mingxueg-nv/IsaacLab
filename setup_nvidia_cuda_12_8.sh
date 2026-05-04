#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   ./setup_nvidia_cuda_12_8.sh -y
#   ./setup_nvidia_cuda_12_8.sh --verify-only
#   ./setup_nvidia_cuda_12_8.sh -y --hold-packages
#   DRIVER_VERSION=580.126.20-0ubuntu0.22.04.1 ./setup_nvidia_cuda_12_8.sh -y
#
# Install the NVIDIA 580.126.20 server driver and CUDA Toolkit 12.8 on
# Ubuntu 22.04. This script intentionally installs cuda-toolkit-12-8 instead
# of the broad cuda/cuda-12-8 meta-package so CUDA does not pull a different
# driver branch.

DRIVER_BRANCH="${DRIVER_BRANCH:-580-server}"
DRIVER_VERSION="${DRIVER_VERSION:-580.126.20-0ubuntu0.22.04.1}"
CUDA_SERIES="${CUDA_SERIES:-12-8}"
CUDA_TOOLKIT_VERSION="${CUDA_TOOLKIT_VERSION:-12.8.2-1}"
CUDA_KEYRING_URL="${CUDA_KEYRING_URL:-https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb}"
CUDA_PROFILE="${CUDA_PROFILE:-/etc/profile.d/cuda-12-8.sh}"

ASSUME_YES=0
VERIFY_ONLY=0
NO_VERIFY=0
HOLD_PACKAGES=0

usage() {
    cat <<EOF
Usage: $0 [options]

Install NVIDIA driver ${DRIVER_VERSION} and CUDA Toolkit ${CUDA_SERIES} on Ubuntu 22.04.

Options:
  -y, --yes          Run without an interactive confirmation prompt.
  --verify-only      Do not install; only verify driver/CUDA state.
  --no-verify        Skip post-install verification.
  --hold-packages    apt-mark hold the installed NVIDIA/CUDA packages.
  -h, --help         Show this help.

Environment overrides:
  DRIVER_BRANCH=${DRIVER_BRANCH}
  DRIVER_VERSION=${DRIVER_VERSION}
  CUDA_SERIES=${CUDA_SERIES}
  CUDA_TOOLKIT_VERSION=${CUDA_TOOLKIT_VERSION}
EOF
}

log() {
    printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

sudo_cmd() {
    if [[ ${EUID} -eq 0 ]]; then
        "$@"
    else
        sudo "$@"
    fi
}

confirm() {
    if [[ ${ASSUME_YES} -eq 1 ]]; then
        return
    fi
    cat <<EOF
This will install system packages with sudo:
  - linux-headers-$(uname -r)
  - nvidia-driver-${DRIVER_BRANCH}=${DRIVER_VERSION}
  - nvidia-dkms-${DRIVER_BRANCH}=${DRIVER_VERSION}
  - cuda-toolkit-${CUDA_SERIES}=${CUDA_TOOLKIT_VERSION}

It may temporarily disrupt GPU access and can require a reboot on some hosts.
EOF
    read -r -p "Continue? [y/N] " answer
    [[ ${answer} =~ ^[Yy]$ ]] || die "Aborted."
}

require_ubuntu_2204() {
    [[ -r /etc/os-release ]] || die "/etc/os-release not found."
    # shellcheck disable=SC1091
    source /etc/os-release
    [[ ${ID:-} == "ubuntu" && ${VERSION_ID:-} == "22.04" ]] || {
        die "This script is for Ubuntu 22.04; found ${PRETTY_NAME:-unknown}."
    }
}

show_gpu_users() {
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        log "Current GPU processes before install"
        nvidia-smi || true
    else
        log "nvidia-smi is not currently working; continuing with package install checks."
    fi
}

install_cuda_repo() {
    log "Installing/updating CUDA APT repository keyring"
    local tmpdir
    tmpdir="$(mktemp -d)"
    trap 'rm -rf "${tmpdir}"' RETURN
    curl -fsSL "${CUDA_KEYRING_URL}" -o "${tmpdir}/cuda-keyring.deb"
    sudo_cmd dpkg -i "${tmpdir}/cuda-keyring.deb"
}

check_package_versions() {
    log "Checking requested package versions are available"
    local missing=0
    for spec in \
        "linux-headers-$(uname -r)" \
        "nvidia-driver-${DRIVER_BRANCH}=${DRIVER_VERSION}" \
        "nvidia-dkms-${DRIVER_BRANCH}=${DRIVER_VERSION}" \
        "cuda-toolkit-${CUDA_SERIES}=${CUDA_TOOLKIT_VERSION}"
    do
        if [[ ${spec} == *=* ]]; then
            local pkg="${spec%%=*}"
            local ver="${spec#*=}"
            if ! apt-cache madison "${pkg}" | awk '{print $3}' | grep -Fxq "${ver}"; then
                echo "Missing package version: ${spec}" >&2
                missing=1
            fi
        else
            if ! apt-cache policy "${spec}" | grep -q 'Candidate:'; then
                echo "Missing package: ${spec}" >&2
                missing=1
            fi
        fi
    done
    [[ ${missing} -eq 0 ]] || die "One or more requested package versions are unavailable."
}

install_packages() {
    log "Updating APT metadata"
    sudo_cmd apt-get update
    check_package_versions

    log "Installing NVIDIA driver and CUDA Toolkit"
    sudo_cmd env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get install -y \
        "linux-headers-$(uname -r)" \
        "nvidia-driver-${DRIVER_BRANCH}=${DRIVER_VERSION}" \
        "nvidia-dkms-${DRIVER_BRANCH}=${DRIVER_VERSION}" \
        "cuda-toolkit-${CUDA_SERIES}=${CUDA_TOOLKIT_VERSION}"
}

write_cuda_profile() {
    log "Writing CUDA environment profile: ${CUDA_PROFILE}"
    sudo_cmd tee "${CUDA_PROFILE}" >/dev/null <<'EOF'
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64
fi
EOF
    sudo_cmd chmod 0644 "${CUDA_PROFILE}"
    sudo_cmd ldconfig
}

load_driver() {
    log "Loading NVIDIA kernel modules"
    sudo_cmd modprobe nvidia || true
    sudo_cmd modprobe nvidia_uvm || true
    sudo_cmd systemctl start nvidia-persistenced 2>/dev/null || true
}

hold_packages() {
    [[ ${HOLD_PACKAGES} -eq 1 ]] || return
    log "Holding NVIDIA/CUDA packages"
    sudo_cmd apt-mark hold \
        "nvidia-driver-${DRIVER_BRANCH}" \
        "nvidia-dkms-${DRIVER_BRANCH}" \
        "cuda-toolkit-${CUDA_SERIES}"
}

verify_install() {
    log "Verifying NVIDIA driver"
    dkms status | grep -E "nvidia.*${DRIVER_VERSION%%-*}" || true
    nvidia-smi

    log "Verifying CUDA Toolkit"
    # shellcheck disable=SC1090
    source "${CUDA_PROFILE}"
    command -v nvcc
    nvcc --version

    log "Installed package versions"
    dpkg-query -W -f='${Package} ${Version}\n' \
        "nvidia-driver-${DRIVER_BRANCH}" \
        "nvidia-dkms-${DRIVER_BRANCH}" \
        "libnvidia-compute-${DRIVER_BRANCH}" \
        "cuda-toolkit-${CUDA_SERIES}" \
        "cuda-nvcc-${CUDA_SERIES}" \
        "linux-headers-$(uname -r)" \
        2>/dev/null || true
}

main() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -y|--yes)
                ASSUME_YES=1
                ;;
            --verify-only)
                VERIFY_ONLY=1
                ;;
            --no-verify)
                NO_VERIFY=1
                ;;
            --hold-packages)
                HOLD_PACKAGES=1
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                die "Unknown option: $1"
                ;;
        esac
        shift
    done

    require_ubuntu_2204

    if [[ ${VERIFY_ONLY} -eq 1 ]]; then
        verify_install
        exit 0
    fi

    confirm
    sudo_cmd -v
    show_gpu_users
    install_cuda_repo
    install_packages
    write_cuda_profile
    load_driver
    hold_packages

    if [[ ${NO_VERIFY} -eq 0 ]]; then
        verify_install
    fi

    log "Done. Open a new login shell, or run: source ${CUDA_PROFILE}"
}

main "$@"
