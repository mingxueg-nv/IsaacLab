#!/usr/bin/env bash
# Start one Cosmos-Transfer2.5 ZMQ service per GPU for RL data augmentation.
#
# Usage:
#   ./start_cosmos_workers.sh [OPTIONS]
#
# Options:
#   --num-gpus N          Number of GPUs to use (default: all available)
#   --base-port PORT      First ZMQ port; subsequent GPUs use PORT+1, PORT+2, ...
#                         (default: 5557)
#   --gpu-offset OFFSET   Start from GPU index OFFSET (default: 0)
#   --control-type TYPE   Cosmos control modality: edge|depth|seg|vis (default: edge)
#   --num-steps N         Diffusion steps; more = higher quality, slower (default: 5)
#   --guidance SCALE      Classifier-free guidance scale (default: 3.0)
#   --hf-home PATH        HF_HOME cache directory (default: /tmp/hf_cache)
#   --log-dir DIR         Directory for per-worker log files (default: ./cosmos_logs)
#   --wait-ready          Block until all workers print "ready" (default: off)
#
# Example (8 GPUs, ports 5557-5564):
#   ./start_cosmos_workers.sh --num-gpus 8 --base-port 5557
#
# To stop all workers:
#   kill $(cat /tmp/cosmos_worker_pids.txt)

set -eo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COSMOS_DIR="${SCRIPT_DIR}/Isaac-GR00T/third_party/cosmos-transfer2.5"
COSMOS_SERVICE="${SCRIPT_DIR}/Isaac-GR00T/scripts/cosmos_service.py"
COSMOS_VENV="${COSMOS_DIR}/.venv/bin/python"
PID_FILE="/tmp/cosmos_worker_pids.txt"

NUM_GPUS=""           # auto-detect if empty
BASE_PORT=5557
GPU_OFFSET=0
CONTROL_TYPE="edge"
NUM_STEPS=5
GUIDANCE=3.0
HF_HOME="${HF_HOME:-/tmp/hf_cache}"
LOG_DIR="${SCRIPT_DIR}/cosmos_logs"
WAIT_READY=0

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-gpus)    NUM_GPUS="$2";    shift 2 ;;
        --base-port)   BASE_PORT="$2";   shift 2 ;;
        --gpu-offset)  GPU_OFFSET="$2";  shift 2 ;;
        --control-type) CONTROL_TYPE="$2"; shift 2 ;;
        --num-steps)   NUM_STEPS="$2";   shift 2 ;;
        --guidance)    GUIDANCE="$2";    shift 2 ;;
        --hf-home)     HF_HOME="$2";     shift 2 ;;
        --log-dir)     LOG_DIR="$2";     shift 2 ;;
        --wait-ready)  WAIT_READY=1;     shift ;;
        -h|--help)
            sed -n '2,30p' "$0" | sed 's/^# //'
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate environment
# ---------------------------------------------------------------------------
if [[ ! -f "${COSMOS_VENV}" ]]; then
    echo "[ERROR] Cosmos venv not found at: ${COSMOS_VENV}" >&2
    echo "        Run: cd ${COSMOS_DIR} && uv sync --extra=cu128" >&2
    exit 1
fi

if [[ ! -f "${COSMOS_SERVICE}" ]]; then
    echo "[ERROR] cosmos_service.py not found at: ${COSMOS_SERVICE}" >&2
    exit 1
fi

# Auto-detect GPU count if not specified
if [[ -z "${NUM_GPUS}" ]]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    if [[ "${NUM_GPUS}" -eq 0 ]]; then
        echo "[ERROR] No GPUs detected. Install nvidia-smi or pass --num-gpus." >&2
        exit 1
    fi
    echo "[INFO] Auto-detected ${NUM_GPUS} GPU(s)"
fi

mkdir -p "${LOG_DIR}"
rm -f "${PID_FILE}"

# ---------------------------------------------------------------------------
# Launch workers
# ---------------------------------------------------------------------------
echo "[INFO] Starting ${NUM_GPUS} Cosmos worker(s)"
echo "[INFO] Control type : ${CONTROL_TYPE}"
echo "[INFO] Diffusion steps: ${NUM_STEPS}"
echo "[INFO] Guidance scale : ${GUIDANCE}"
echo "[INFO] Base port      : ${BASE_PORT}"
echo "[INFO] GPU offset     : ${GPU_OFFSET}"
echo "[INFO] HF_HOME        : ${HF_HOME}"
echo "[INFO] Log directory  : ${LOG_DIR}"
echo ""

PIDS=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID=$((GPU_OFFSET + i))
    PORT=$((BASE_PORT + i))
    LOG_FILE="${LOG_DIR}/cosmos_worker_gpu${GPU_ID}_port${PORT}.log"

    echo "[INFO] GPU ${GPU_ID} → port ${PORT} → ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    HF_HOME="${HF_HOME}" \
    PATH="${COSMOS_DIR}/.venv/bin:${PATH}" \
        "${COSMOS_VENV}" -u "${COSMOS_SERVICE}" \
            --port "${PORT}" \
            --control-type "${CONTROL_TYPE}" \
            --num-steps "${NUM_STEPS}" \
            --guidance "${GUIDANCE}" \
        > "${LOG_FILE}" 2>&1 &

    PID=$!
    PIDS+=("${PID}")
    echo "${PID}" >> "${PID_FILE}"
done

echo ""
echo "[INFO] All ${NUM_GPUS} worker(s) launched. PIDs saved to ${PID_FILE}"
echo "[INFO] To stop all workers: kill \$(cat ${PID_FILE})"
echo ""

# ---------------------------------------------------------------------------
# Optionally wait until all workers print "ready"
# ---------------------------------------------------------------------------
if [[ "${WAIT_READY}" -eq 1 ]]; then
    echo "[INFO] Waiting for all workers to be ready..."
    READY_COUNT=0
    TIMEOUT=300  # 5 minutes max
    ELAPSED=0
    while [[ "${READY_COUNT}" -lt "${NUM_GPUS}" && "${ELAPSED}" -lt "${TIMEOUT}" ]]; do
        READY_COUNT=0
        for i in $(seq 0 $((NUM_GPUS - 1))); do
            GPU_ID=$((GPU_OFFSET + i))
            PORT=$((BASE_PORT + i))
            LOG_FILE="${LOG_DIR}/cosmos_worker_gpu${GPU_ID}_port${PORT}.log"
            if grep -q "ready on port" "${LOG_FILE}" 2>/dev/null; then
                READY_COUNT=$((READY_COUNT + 1))
            fi
        done
        if [[ "${READY_COUNT}" -lt "${NUM_GPUS}" ]]; then
            sleep 5
            ELAPSED=$((ELAPSED + 5))
            echo "[INFO] ${READY_COUNT}/${NUM_GPUS} ready (${ELAPSED}s elapsed)..."
        fi
    done

    if [[ "${READY_COUNT}" -eq "${NUM_GPUS}" ]]; then
        echo "[INFO] All ${NUM_GPUS} Cosmos worker(s) ready!"
    else
        echo "[WARN] Timed out after ${TIMEOUT}s. Only ${READY_COUNT}/${NUM_GPUS} workers ready."
        echo "       Check logs in ${LOG_DIR}/ for errors."
    fi
else
    echo "[INFO] Workers starting in background. Monitor with:"
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        GPU_ID=$((GPU_OFFSET + i))
        PORT=$((BASE_PORT + i))
        echo "         tail -f ${LOG_DIR}/cosmos_worker_gpu${GPU_ID}_port${PORT}.log"
    done
fi
