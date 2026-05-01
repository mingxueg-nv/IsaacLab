#!/usr/bin/env bash
# Start one Z-Image-Fun ZMQ service per GPU for RL data augmentation.
#
# Usage:
#   ./start_z_image_workers.sh [OPTIONS] [-- EXTRA_Z_IMAGE_SERVICE_ARGS...]
#
# Options:
#   --num-gpus N          Number of GPUs to use (default: all available)
#   --base-port PORT      First ZMQ port; subsequent GPUs use PORT+1, PORT+2, ...
#                         (default: 5657)
#   --gpu-offset OFFSET   Start from GPU index OFFSET (default: 0)
#   --num-steps N         Diffusion steps (default: 2)
#   --guidance SCALE      Classifier-free guidance scale (default: 1.0)
#   --second-sigma VALUE  Second sigma for the default 2-step schedule (default: 0.80)
#   --lora-weight VALUE   Z-Image-Fun LoRA distill weight (default: 1.0)
#   --memory-mode MODE    model_full_load|model_full_load_and_qfloat8|model_cpu_offload|
#                         model_cpu_offload_and_qfloat8|sequential_cpu_offload
#                         (default: model_full_load)
#   --fp16                Use float16 instead of bfloat16
#   --compile             torch.compile transformer layers and VAE modules
#   --config-path PATH    Override VideoX-Fun Z-Image config path
#   --model-name PATH     Override base Z-Image model folder
#   --transformer-path PATH
#                         Override Z-Image-Fun ControlNet checkpoint path
#   --lora-path PATH      Override LoRA checkpoint path. Pass '' to disable LoRA.
#   --prompt TEXT         Override normal control prompt
#   --negative-prompt TEXT
#                         Override normal negative prompt
#   --inpaint-prompt TEXT Override inpaint prompt. If omitted, the service uses
#                         seed-selected default background variants.
#   --inpaint-negative-prompt TEXT
#                         Override inpaint negative prompt
#   --hf-home PATH        HF_HOME cache directory (default: /tmp/hf_cache)
#   --log-dir DIR         Directory for per-worker log files (default: ./z_image_logs)
#   --wait-ready          Block until all workers print "ready" (default: off)
#   --ready-timeout SEC   Max seconds to wait for readiness (default: 900)
#   --warmup              After readiness, send one dummy batched inpaint/control
#                         request to each worker. Implies --wait-ready.
#   --warmup-batch-size N Batch size for warmup request (default: 8)
#   --warmup-time-steps N Time dimension for warmup request (default: 1)
#   --warmup-height PX    Warmup image height (default: 448)
#   --warmup-width PX     Warmup image width (default: 448)
#   --warmup-seed SEED    Base seed for warmup requests (default: 20260430)
#   --warmup-timeout SEC  Per-worker warmup timeout (default: 900)
#
# Example (8 GPUs, ports 5657-5664):
#   ./start_z_image_workers.sh --num-gpus 8 --base-port 5657 --wait-ready --warmup
#
# Example with extra service args:
#   ./start_z_image_workers.sh --num-gpus 1 -- --sigmas 1.0 0.65
#
# To stop all workers:
#   kill $(cat /tmp/z_image_worker_pids.txt)

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROOT_DIR="${SCRIPT_DIR}/Isaac-GR00T"
VIDEOX_FUN_DIR="${GROOT_DIR}/third_party/VideoX-Fun"
Z_IMAGE_SERVICE="${GROOT_DIR}/scripts/z_image_service.py"
Z_IMAGE_WARMUP="${GROOT_DIR}/scripts/warmup_z_image_service.py"
Z_IMAGE_VENV="${VIDEOX_FUN_DIR}/.venv/bin/python"
PID_FILE="/tmp/z_image_worker_pids.txt"

NUM_GPUS=""           # auto-detect if empty
BASE_PORT=5657
GPU_OFFSET=0
NUM_STEPS=2
GUIDANCE=1.0
SECOND_SIGMA=0.80
LORA_WEIGHT=1.0
MEMORY_MODE="model_full_load"
FP16=0
COMPILE=1
HF_HOME="${HF_HOME:-/tmp/hf_cache}"
LOG_DIR="${SCRIPT_DIR}/z_image_logs"
WAIT_READY=0
READY_TIMEOUT=900
WARMUP=0
WARMUP_BATCH_SIZE=8
WARMUP_TIME_STEPS=1
WARMUP_HEIGHT=448
WARMUP_WIDTH=448
WARMUP_SEED=20260430
WARMUP_TIMEOUT=900

CONFIG_PATH=""
MODEL_NAME=""
TRANSFORMER_PATH=""
LORA_PATH=""
PROMPT=""
NEGATIVE_PROMPT=""
INPAINT_PROMPT=""
INPAINT_NEGATIVE_PROMPT=""
HAVE_LORA_PATH=0
EXTRA_ARGS=()

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-gpus)                 NUM_GPUS="$2";                 shift 2 ;;
        --base-port)                BASE_PORT="$2";                shift 2 ;;
        --gpu-offset)               GPU_OFFSET="$2";               shift 2 ;;
        --num-steps)                NUM_STEPS="$2";                shift 2 ;;
        --guidance)                 GUIDANCE="$2";                 shift 2 ;;
        --second-sigma)             SECOND_SIGMA="$2";             shift 2 ;;
        --lora-weight)              LORA_WEIGHT="$2";              shift 2 ;;
        --memory-mode)              MEMORY_MODE="$2";              shift 2 ;;
        --fp16)                     FP16=1;                        shift ;;
        --compile)                  COMPILE=1;                     shift ;;
        --config-path)              CONFIG_PATH="$2";              shift 2 ;;
        --model-name)               MODEL_NAME="$2";               shift 2 ;;
        --transformer-path)         TRANSFORMER_PATH="$2";         shift 2 ;;
        --lora-path)                LORA_PATH="$2"; HAVE_LORA_PATH=1; shift 2 ;;
        --prompt)                   PROMPT="$2";                   shift 2 ;;
        --negative-prompt)          NEGATIVE_PROMPT="$2";          shift 2 ;;
        --inpaint-prompt)           INPAINT_PROMPT="$2";           shift 2 ;;
        --inpaint-negative-prompt)  INPAINT_NEGATIVE_PROMPT="$2";  shift 2 ;;
        --hf-home)                  HF_HOME="$2";                  shift 2 ;;
        --log-dir)                  LOG_DIR="$2";                  shift 2 ;;
        --wait-ready)               WAIT_READY=1;                  shift ;;
        --ready-timeout)            READY_TIMEOUT="$2";            shift 2 ;;
        --warmup)                   WARMUP=1; WAIT_READY=1;       shift ;;
        --warmup-batch-size)        WARMUP_BATCH_SIZE="$2";        shift 2 ;;
        --warmup-time-steps)        WARMUP_TIME_STEPS="$2";        shift 2 ;;
        --warmup-height)            WARMUP_HEIGHT="$2";            shift 2 ;;
        --warmup-width)             WARMUP_WIDTH="$2";             shift 2 ;;
        --warmup-seed)              WARMUP_SEED="$2";              shift 2 ;;
        --warmup-timeout)           WARMUP_TIMEOUT="$2";           shift 2 ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        -h|--help)
            sed -n '2,45p' "$0" | sed 's/^# //'
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate environment
# ---------------------------------------------------------------------------
if [[ ! -x "${Z_IMAGE_VENV}" ]]; then
    echo "[ERROR] Z-Image venv not found at: ${Z_IMAGE_VENV}" >&2
    echo "        Run: cd ${GROOT_DIR} && bash scripts/setup_z_image_venv.sh" >&2
    exit 1
fi

if [[ ! -f "${Z_IMAGE_SERVICE}" ]]; then
    echo "[ERROR] z_image_service.py not found at: ${Z_IMAGE_SERVICE}" >&2
    exit 1
fi

if [[ "${WARMUP}" -eq 1 && ! -f "${Z_IMAGE_WARMUP}" ]]; then
    echo "[ERROR] warmup_z_image_service.py not found at: ${Z_IMAGE_WARMUP}" >&2
    exit 1
fi

if [[ ! -d "${VIDEOX_FUN_DIR}" ]]; then
    echo "[ERROR] VideoX-Fun directory not found at: ${VIDEOX_FUN_DIR}" >&2
    exit 1
fi

# Auto-detect GPU count if not specified.
if [[ -z "${NUM_GPUS}" ]]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
    if [[ "${NUM_GPUS}" -eq 0 ]]; then
        echo "[ERROR] No GPUs detected. Install nvidia-smi or pass --num-gpus." >&2
        exit 1
    fi
    echo "[INFO] Auto-detected ${NUM_GPUS} GPU(s)"
fi

mkdir -p "${LOG_DIR}"
rm -f "${PID_FILE}"

COMMON_ARGS=(
    --num-steps "${NUM_STEPS}"
    --guidance "${GUIDANCE}"
    --second-sigma "${SECOND_SIGMA}"
    --lora-weight "${LORA_WEIGHT}"
    --memory-mode "${MEMORY_MODE}"
)

if [[ "${FP16}" -eq 1 ]]; then
    COMMON_ARGS+=(--fp16)
fi
if [[ "${COMPILE}" -eq 1 ]]; then
    COMMON_ARGS+=(--compile)
fi
if [[ -n "${CONFIG_PATH}" ]]; then
    COMMON_ARGS+=(--config-path "${CONFIG_PATH}")
fi
if [[ -n "${MODEL_NAME}" ]]; then
    COMMON_ARGS+=(--model-name "${MODEL_NAME}")
fi
if [[ -n "${TRANSFORMER_PATH}" ]]; then
    COMMON_ARGS+=(--transformer-path "${TRANSFORMER_PATH}")
fi
if [[ "${HAVE_LORA_PATH}" -eq 1 ]]; then
    COMMON_ARGS+=(--lora-path "${LORA_PATH}")
fi
if [[ -n "${PROMPT}" ]]; then
    COMMON_ARGS+=(--prompt "${PROMPT}")
fi
if [[ -n "${NEGATIVE_PROMPT}" ]]; then
    COMMON_ARGS+=(--negative-prompt "${NEGATIVE_PROMPT}")
fi
if [[ -n "${INPAINT_PROMPT}" ]]; then
    COMMON_ARGS+=(--inpaint-prompt "${INPAINT_PROMPT}")
fi
if [[ -n "${INPAINT_NEGATIVE_PROMPT}" ]]; then
    COMMON_ARGS+=(--inpaint-negative-prompt "${INPAINT_NEGATIVE_PROMPT}")
fi
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
    COMMON_ARGS+=("${EXTRA_ARGS[@]}")
fi

# ---------------------------------------------------------------------------
# Launch workers
# ---------------------------------------------------------------------------
echo "[INFO] Starting ${NUM_GPUS} Z-Image worker(s)"
echo "[INFO] Diffusion steps: ${NUM_STEPS}"
echo "[INFO] Guidance scale : ${GUIDANCE}"
echo "[INFO] Second sigma   : ${SECOND_SIGMA}"
echo "[INFO] LoRA weight    : ${LORA_WEIGHT}"
echo "[INFO] Memory mode    : ${MEMORY_MODE}"
echo "[INFO] Base port      : ${BASE_PORT}"
echo "[INFO] GPU offset     : ${GPU_OFFSET}"
echo "[INFO] HF_HOME        : ${HF_HOME}"
echo "[INFO] Log directory  : ${LOG_DIR}"
if [[ "${WARMUP}" -eq 1 ]]; then
    echo "[INFO] Warmup         : enabled (B=${WARMUP_BATCH_SIZE}, T=${WARMUP_TIME_STEPS}, ${WARMUP_HEIGHT}x${WARMUP_WIDTH})"
else
    echo "[INFO] Warmup         : disabled"
fi
echo ""

PIDS=()
for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID=$((GPU_OFFSET + i))
    PORT=$((BASE_PORT + i))
    LOG_FILE="${LOG_DIR}/z_image_worker_gpu${GPU_ID}_port${PORT}.log"

    echo "[INFO] GPU ${GPU_ID} -> port ${PORT} -> ${LOG_FILE}"

    (
        cd "${VIDEOX_FUN_DIR}"
        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        HF_HOME="${HF_HOME}" \
        "${Z_IMAGE_VENV}" -u "${Z_IMAGE_SERVICE}" \
            --port "${PORT}" \
            "${COMMON_ARGS[@]}"
    ) > "${LOG_FILE}" 2>&1 &

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
    ELAPSED=0
    while [[ "${READY_COUNT}" -lt "${NUM_GPUS}" && "${ELAPSED}" -lt "${READY_TIMEOUT}" ]]; do
        READY_COUNT=0
        for i in $(seq 0 $((NUM_GPUS - 1))); do
            GPU_ID=$((GPU_OFFSET + i))
            PORT=$((BASE_PORT + i))
            LOG_FILE="${LOG_DIR}/z_image_worker_gpu${GPU_ID}_port${PORT}.log"
            PID="${PIDS[$i]}"
            if grep -q "Z-Image service ready on port ${PORT}" "${LOG_FILE}" 2>/dev/null; then
                READY_COUNT=$((READY_COUNT + 1))
            elif ! kill -0 "${PID}" 2>/dev/null; then
                echo "[ERROR] Worker on GPU ${GPU_ID}, port ${PORT} exited before becoming ready." >&2
                echo "        Log: ${LOG_FILE}" >&2
                tail -80 "${LOG_FILE}" >&2 || true
                exit 1
            fi
        done
        if [[ "${READY_COUNT}" -lt "${NUM_GPUS}" ]]; then
            sleep 5
            ELAPSED=$((ELAPSED + 5))
            echo "[INFO] ${READY_COUNT}/${NUM_GPUS} ready (${ELAPSED}s elapsed)..."
        fi
    done

    if [[ "${READY_COUNT}" -eq "${NUM_GPUS}" ]]; then
        echo "[INFO] All ${NUM_GPUS} Z-Image worker(s) ready!"
    else
        echo "[WARN] Timed out after ${READY_TIMEOUT}s. Only ${READY_COUNT}/${NUM_GPUS} workers ready."
        echo "       Check logs in ${LOG_DIR}/ for errors."
        if [[ "${WARMUP}" -eq 1 ]]; then
            echo "[ERROR] Cannot warm up until all workers are ready." >&2
            exit 1
        fi
    fi
else
    echo "[INFO] Workers starting in background. Monitor with:"
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        GPU_ID=$((GPU_OFFSET + i))
        PORT=$((BASE_PORT + i))
        echo "         tail -f ${LOG_DIR}/z_image_worker_gpu${GPU_ID}_port${PORT}.log"
    done
fi

# ---------------------------------------------------------------------------
# Optionally warm up the batched Z-Image inpaint/control path
# ---------------------------------------------------------------------------
if [[ "${WARMUP}" -eq 1 ]]; then
    echo ""
    echo "[INFO] Warming up ${NUM_GPUS} Z-Image worker(s) with batched dummy requests..."
    WARMUP_PIDS=()
    WARMUP_LOGS=()
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        GPU_ID=$((GPU_OFFSET + i))
        PORT=$((BASE_PORT + i))
        WARMUP_LOG="${LOG_DIR}/z_image_warmup_gpu${GPU_ID}_port${PORT}.log"
        WARMUP_LOGS+=("${WARMUP_LOG}")
        echo "[INFO] Warmup GPU ${GPU_ID}, port ${PORT} -> ${WARMUP_LOG}"
        (
            "${Z_IMAGE_VENV}" "${Z_IMAGE_WARMUP}" \
                --host localhost \
                --ports "${PORT}" \
                --batch-size "${WARMUP_BATCH_SIZE}" \
                --time-steps "${WARMUP_TIME_STEPS}" \
                --height "${WARMUP_HEIGHT}" \
                --width "${WARMUP_WIDTH}" \
                --seed "$((WARMUP_SEED + i * 1000003))" \
                --timeout-s "${WARMUP_TIMEOUT}"
        ) > "${WARMUP_LOG}" 2>&1 &
        WARMUP_PIDS+=("$!")
    done

    WARMUP_FAILED=0
    for i in "${!WARMUP_PIDS[@]}"; do
        if ! wait "${WARMUP_PIDS[$i]}"; then
            WARMUP_FAILED=1
            echo "[ERROR] Warmup failed. Log: ${WARMUP_LOGS[$i]}" >&2
            tail -80 "${WARMUP_LOGS[$i]}" >&2 || true
        fi
    done

    if [[ "${WARMUP_FAILED}" -ne 0 ]]; then
        exit 1
    fi

    echo "[INFO] Warmup complete."
    for log_file in "${WARMUP_LOGS[@]}"; do
        sed 's/^/[INFO]   /' "${log_file}"
    done
fi
