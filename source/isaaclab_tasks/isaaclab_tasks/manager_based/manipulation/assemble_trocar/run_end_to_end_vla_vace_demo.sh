#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_ROOT="${ISAACLAB_ROOT:-/localhome/local-pengfeig/pengfeig/IsaacLab}"
SCOPE_ROOT="${SCOPE_ROOT:-/localhome/local-pengfeig/pengfeig/scope}"
MODEL_PATH="${MODEL_PATH:-/localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps_new_trocar_460samples}"
SCOPE_MODELS_DIR="${SCOPE_MODELS_DIR:-/localhome/local-pengfeig/pengfeig/models/scope}"
OUT_DIR="${OUT_DIR:-/tmp/vla_vace_live_demo}"
NUM_FRAMES="${NUM_FRAMES:-48}"
FPS="${FPS:-12}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
VACE_WIDTH="${VACE_WIDTH:-$CAMERA_WIDTH}"
VACE_HEIGHT="${VACE_HEIGHT:-$CAMERA_HEIGHT}"
VACE_CHUNKS="${VACE_CHUNKS:-4}"
FRAMES_PER_CHUNK="${FRAMES_PER_CHUNK:-12}"
OVERLAP_FRAMES="${OVERLAP_FRAMES:-0}"
MASK_DILATE_PX="${MASK_DILATE_PX:-0}"
MASK_FILL_HOLES="${MASK_FILL_HOLES:-0}"
MASK_CLOSE_PX="${MASK_CLOSE_PX:-5}"
MASK_TEMPORAL_RADIUS="${MASK_TEMPORAL_RADIUS:-1}"
MASK_GUARD_PX="${MASK_GUARD_PX:-2}"
VACE_COMPOSITE_MODE="${VACE_COMPOSITE_MODE:-upperroom}"
UPPERROOM_Y_CUT="${UPPERROOM_Y_CUT:-155}"
UPPERROOM_Y_FADE="${UPPERROOM_Y_FADE:-45}"
UPPERROOM_DISTANCE_START="${UPPERROOM_DISTANCE_START:-8}"
UPPERROOM_DISTANCE_END="${UPPERROOM_DISTANCE_END:-20}"
RENDER_ANTIALIASING="${RENDER_ANTIALIASING:-DLAA}"
VACE_CONTEXT_SCALE="${VACE_CONTEXT_SCALE:-1.5}"
VACE_WARMUP_CHUNKS="${VACE_WARMUP_CHUNKS:-1}"
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

cd "$SCOPE_ROOT"
python -m scope.core.pipelines.longlive.isaac_vace_background_inpaint_demo \
  --input_video "$ISAAC_OUT/front_camera_rgb.mp4" \
  --mask_video "$ISAAC_OUT/front_camera_background_mask.mp4" \
  --depth_video "$ISAAC_OUT/front_camera_depth_inverse.mp4" \
  --output_dir "$VACE_OUT" \
  --num_chunks "$VACE_CHUNKS" \
  --frames_per_chunk "$FRAMES_PER_CHUNK" \
  --overlap_frames "$OVERLAP_FRAMES" \
  --height "$VACE_HEIGHT" \
  --width "$VACE_WIDTH" \
  --fps "$FPS" \
  --vace_context_scale "$VACE_CONTEXT_SCALE" \
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
