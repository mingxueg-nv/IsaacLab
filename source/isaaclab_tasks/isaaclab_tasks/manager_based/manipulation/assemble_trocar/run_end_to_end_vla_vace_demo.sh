#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_ROOT="${ISAACLAB_ROOT:-/localhome/local-pengfeig/pengfeig/IsaacLab}"
SCOPE_ROOT="${SCOPE_ROOT:-/localhome/local-pengfeig/pengfeig/scope}"
MODEL_PATH="${MODEL_PATH:-/localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps_new_trocar_460samples}"
SCOPE_MODELS_DIR="${SCOPE_MODELS_DIR:-/localhome/local-pengfeig/pengfeig/models/scope}"
OUT_DIR="${OUT_DIR:-/tmp/vla_vace_live_demo}"
NUM_FRAMES="${NUM_FRAMES:-48}"
FPS="${FPS:-12}"
CAMERA_WIDTH="${CAMERA_WIDTH:-320}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-240}"
VACE_WIDTH="${VACE_WIDTH:-512}"
VACE_HEIGHT="${VACE_HEIGHT:-384}"
FRAMES_PER_CHUNK="${FRAMES_PER_CHUNK:-12}"
OVERLAP_FRAMES="${OVERLAP_FRAMES:-0}"
COMMIT_FRAMES_PER_CHUNK="${COMMIT_FRAMES_PER_CHUNK:-6}"
if [[ -z "${VACE_CHUNKS:-}" ]]; then
  if [[ "$COMMIT_FRAMES_PER_CHUNK" -gt 0 ]]; then
    VACE_CHUNKS=$(( (NUM_FRAMES + COMMIT_FRAMES_PER_CHUNK - 1) / COMMIT_FRAMES_PER_CHUNK ))
  else
    VACE_CHUNKS=$(( (NUM_FRAMES + FRAMES_PER_CHUNK - OVERLAP_FRAMES - 1) / (FRAMES_PER_CHUNK - OVERLAP_FRAMES) ))
  fi
fi
MASK_DILATE_PX="${MASK_DILATE_PX:-0}"
MASK_FILL_HOLES="${MASK_FILL_HOLES:-0}"
MASK_CLOSE_PX="${MASK_CLOSE_PX:-5}"
MASK_TEMPORAL_RADIUS="${MASK_TEMPORAL_RADIUS:-1}"
MASK_GUARD_PX="${MASK_GUARD_PX:-4}"
VACE_COMPOSITE_MODE="${VACE_COMPOSITE_MODE:-upperroom}"
UPPERROOM_Y_CUT="${UPPERROOM_Y_CUT:-155}"
UPPERROOM_Y_FADE="${UPPERROOM_Y_FADE:-45}"
UPPERROOM_DISTANCE_START="${UPPERROOM_DISTANCE_START:-8}"
UPPERROOM_DISTANCE_END="${UPPERROOM_DISTANCE_END:-20}"
RENDER_ANTIALIASING="${RENDER_ANTIALIASING:-DLAA}"
VACE_CONTEXT_SCALE="${VACE_CONTEXT_SCALE:-1.5}"
VACE_WARMUP_CHUNKS="${VACE_WARMUP_CHUNKS:-1}"
VACE_DENOISING_STEPS="${VACE_DENOISING_STEPS:-1000,500}"
FIXED_DELAY_CHUNKS="${FIXED_DELAY_CHUNKS:-2}"
LIVE_FRAME_PERIOD_MS="${LIVE_FRAME_PERIOD_MS:-0}"
COMPOSITE_ALPHA_ERODE_PX="${COMPOSITE_ALPHA_ERODE_PX:-1}"
COMPOSITE_ALPHA_BLUR_PX="${COMPOSITE_ALPHA_BLUR_PX:-7}"
COMPOSITE_ALPHA_TEMPORAL_RADIUS="${COMPOSITE_ALPHA_TEMPORAL_RADIUS:-1}"
VACE_PROMPT="${VACE_PROMPT:-Single camera view. Render a realistic sterile operating room background with clean walls, ceiling lights, surgical lamp positions, monitor stands, cabinets, IV poles, drapes, floor material, reflections, background equipment behind the workspace, moderate clutter, and neutral clinical illumination. Keep the background physically consistent with camera perspective and depth. The masked region must contain only background surfaces and distant room equipment, with no new foreground subject. Existing robot hands, black grippers, surgical tray, trocar tools, and task objects are foreground copied from the source video; do not generate, duplicate, or extend them in the masked background.}"

ASSEMBLE_DIR="$ISAACLAB_ROOT/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar"
ISAAC_OUT="$OUT_DIR/isaac"
VACE_OUT="$OUT_DIR/vace"
SIDE_BY_SIDE="$OUT_DIR/vla_raw_vs_vace.mp4"

mkdir -p "$OUT_DIR"

source /localhome/local-pengfeig/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab_develop_6.0_pg

export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export DAYDREAM_SCOPE_MODELS_DIR="$SCOPE_MODELS_DIR"

cd "$ISAACLAB_ROOT"
./isaaclab.sh -p "$ASSEMBLE_DIR/run_gr00t_isaac_demo_stream.py" \
  --model_path "$MODEL_PATH" \
  --config_dir "$ASSEMBLE_DIR/config" \
  --num_envs 1 \
  --num_frames "$NUM_FRAMES" \
  --warmup_steps 4 \
  --camera front_camera \
  --camera_width "$CAMERA_WIDTH" \
  --camera_height "$CAMERA_HEIGHT" \
  --fps "$FPS" \
  --output_dir "$ISAAC_OUT" \
  --background_mask \
  --mask_dilate_px "$MASK_DILATE_PX" \
  --headless \
  --enable_cameras \
  --rendering_mode balanced \
  --render_antialiasing "$RENDER_ANTIALIASING" \
  --render_translucency off

if [[ "$LIVE_FRAME_PERIOD_MS" == "0" || "$LIVE_FRAME_PERIOD_MS" == "0.0" ]]; then
  LIVE_FRAME_PERIOD_MS="$(
    python - <<PY
import json
from pathlib import Path
metrics = json.loads(Path("$ISAAC_OUT/metrics.json").read_text())
print(metrics.get("total_frame_ms", {}).get("median_ms", 0.0))
PY
  )"
fi
echo "Using LIVE_FRAME_PERIOD_MS=${LIVE_FRAME_PERIOD_MS} for fixed-delay metrics."

cd "$SCOPE_ROOT"
python -m scope.core.pipelines.longlive.isaac_vace_background_inpaint_demo \
  --input_video "$ISAAC_OUT/front_camera_rgb.mp4" \
  --mask_video "$ISAAC_OUT/front_camera_background_mask.mp4" \
  --depth_video "$ISAAC_OUT/front_camera_depth_inverse.mp4" \
  --output_dir "$VACE_OUT" \
  --num_chunks "$VACE_CHUNKS" \
  --frames_per_chunk "$FRAMES_PER_CHUNK" \
  --overlap_frames "$OVERLAP_FRAMES" \
  --commit_frames_per_chunk "$COMMIT_FRAMES_PER_CHUNK" \
  --height "$VACE_HEIGHT" \
  --width "$VACE_WIDTH" \
  --fps "$FPS" \
  --vace_context_scale "$VACE_CONTEXT_SCALE" \
  --denoising_steps "$VACE_DENOISING_STEPS" \
  --mask_fill_holes "$MASK_FILL_HOLES" \
  --mask_close_px "$MASK_CLOSE_PX" \
  --mask_temporal_radius "$MASK_TEMPORAL_RADIUS" \
  --mask_guard_px "$MASK_GUARD_PX" \
  --composite_mode "$VACE_COMPOSITE_MODE" \
  --upperroom_y_cut "$UPPERROOM_Y_CUT" \
  --upperroom_y_fade "$UPPERROOM_Y_FADE" \
  --upperroom_distance_start "$UPPERROOM_DISTANCE_START" \
  --upperroom_distance_end "$UPPERROOM_DISTANCE_END" \
  --warmup_chunks "$VACE_WARMUP_CHUNKS" \
  --fixed_delay_chunks "$FIXED_DELAY_CHUNKS" \
  --live_frame_period_ms "$LIVE_FRAME_PERIOD_MS" \
  --composite_alpha_erode_px "$COMPOSITE_ALPHA_ERODE_PX" \
  --composite_alpha_blur_px "$COMPOSITE_ALPHA_BLUR_PX" \
  --composite_alpha_temporal_radius "$COMPOSITE_ALPHA_TEMPORAL_RADIUS" \
  --prompt "$VACE_PROMPT"

ffmpeg -y \
  -i "$ISAAC_OUT/front_camera_rgb.mp4" \
  -i "$VACE_OUT/vace_background_inpaint.mp4" \
  -filter_complex "[0:v]scale=${VACE_WIDTH}:${VACE_HEIGHT},setpts=PTS-STARTPTS[left];[1:v]scale=${VACE_WIDTH}:${VACE_HEIGHT},setpts=PTS-STARTPTS[right];[left][right]hstack=inputs=2:shortest=1[v]" \
  -map "[v]" \
  -c:v libx264 \
  -pix_fmt yuv420p \
  "$SIDE_BY_SIDE"

python - <<PY
from pathlib import Path
print("Demo outputs:")
for path in [
    Path("$ISAAC_OUT/front_camera_rgb.mp4"),
    Path("$ISAAC_OUT/front_camera_depth_inverse.mp4"),
    Path("$ISAAC_OUT/front_camera_background_mask.mp4"),
    Path("$VACE_OUT/vace_background_inpaint.mp4"),
    Path("$VACE_OUT/vace_background_inpaint_raw.mp4"),
    Path("$SIDE_BY_SIDE"),
]:
    print(f"  {path}")
PY
