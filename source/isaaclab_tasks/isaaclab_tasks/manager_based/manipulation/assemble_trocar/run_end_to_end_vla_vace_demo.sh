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
VACE_WIDTH="${VACE_WIDTH:-640}"
VACE_HEIGHT="${VACE_HEIGHT:-480}"
USE_VACE_DEPTH="${USE_VACE_DEPTH:-1}"
VACE_REF_IMAGES="${VACE_REF_IMAGES:-}"
VACE_REF_IMAGES_EVERY_CHUNK="${VACE_REF_IMAGES_EVERY_CHUNK:-1}"
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
MASK_PRESERVE_CLASSES="${MASK_PRESERVE_CLASSES:-robot,tray,cart,trocar,trocar_device}"
MASK_PRESERVE_BLUE_TABLE="${MASK_PRESERVE_BLUE_TABLE:-0}"
MASK_BLUE_TABLE_MIN_BLUE="${MASK_BLUE_TABLE_MIN_BLUE:-95}"
MASK_BLUE_TABLE_MIN_GREEN="${MASK_BLUE_TABLE_MIN_GREEN:-70}"
MASK_BLUE_TABLE_MAX_RED="${MASK_BLUE_TABLE_MAX_RED:-135}"
MASK_BLUE_TABLE_MIN_BLUE_MINUS_RED="${MASK_BLUE_TABLE_MIN_BLUE_MINUS_RED:-25}"
MASK_BLUE_TABLE_MIN_GREEN_MINUS_RED="${MASK_BLUE_TABLE_MIN_GREEN_MINUS_RED:-5}"
MASK_BLUE_TABLE_MIN_X_FRAC="${MASK_BLUE_TABLE_MIN_X_FRAC:-0.48}"
MASK_BLUE_TABLE_MIN_Y_FRAC="${MASK_BLUE_TABLE_MIN_Y_FRAC:-0.30}"
MASK_FILL_HOLES="${MASK_FILL_HOLES:-0}"
MASK_CLOSE_PX="${MASK_CLOSE_PX:-0}"
MASK_TEMPORAL_RADIUS="${MASK_TEMPORAL_RADIUS:-0}"
MASK_GUARD_PX="${MASK_GUARD_PX:-0}"
VACE_COMPOSITE_MODE="${VACE_COMPOSITE_MODE:-mask}"
VACE_PRESENTATION_OUTPUT="${VACE_PRESENTATION_OUTPUT:-raw}"
UPPERROOM_Y_CUT="${UPPERROOM_Y_CUT:-155}"
UPPERROOM_Y_FADE="${UPPERROOM_Y_FADE:-45}"
UPPERROOM_DISTANCE_START="${UPPERROOM_DISTANCE_START:-8}"
UPPERROOM_DISTANCE_END="${UPPERROOM_DISTANCE_END:-20}"
RENDER_ANTIALIASING="${RENDER_ANTIALIASING:-DLAA}"
VACE_CONTEXT_SCALE="${VACE_CONTEXT_SCALE:-1.5}"
VACE_WARMUP_CHUNKS="${VACE_WARMUP_CHUNKS:-1}"
VACE_DENOISING_STEPS="${VACE_DENOISING_STEPS:-1000,500}"
VACE_HISTORY_CONDITIONING="${VACE_HISTORY_CONDITIONING:-none}"
FIXED_DELAY_CHUNKS="${FIXED_DELAY_CHUNKS:-2}"
LIVE_FRAME_PERIOD_MS="${LIVE_FRAME_PERIOD_MS:-0}"
COMPOSITE_ALPHA_ERODE_PX="${COMPOSITE_ALPHA_ERODE_PX:-0}"
COMPOSITE_ALPHA_BLUR_PX="${COMPOSITE_ALPHA_BLUR_PX:-1}"
COMPOSITE_ALPHA_TEMPORAL_RADIUS="${COMPOSITE_ALPHA_TEMPORAL_RADIUS:-1}"
VACE_PROMPT_VARIANT="${VACE_PROMPT_VARIANT:-0}"
VACE_PROMPT_LAYOUT="${VACE_PROMPT_LAYOUT:-single}"

SOURCE_FOUR_PANEL_LAYOUT_PROMPT="A four-panel multi-view scene. Treat each panel as an independent camera view: top-left is the left camera, top-right is the right camera, bottom-left is a tabletop view of the workspace, and bottom-right is empty black padding. Keep this grid layout unchanged and leave the empty black panel black. "
SINGLE_CAMERA_LAYOUT_PROMPT="A single front-camera tabletop view of the workspace. "
BK_INSTRUCTION_PROMPT="Keep the background physically consistent with the camera perspective and depth. The masked region must contain only background surfaces and distant room equipment, with no new foreground subject. "

case "$VACE_PROMPT_LAYOUT" in
  single)
    VACE_PROMPT_LAYOUT_PREFIX="$SINGLE_CAMERA_LAYOUT_PROMPT"
    ;;
  source_four_panel)
    VACE_PROMPT_LAYOUT_PREFIX="$SOURCE_FOUR_PANEL_LAYOUT_PROMPT"
    ;;
  *)
    echo "VACE_PROMPT_LAYOUT must be 'single' or 'source_four_panel'; got '$VACE_PROMPT_LAYOUT'." >&2
    exit 2
    ;;
esac

DEFAULT_BACKGROUND_INPAINT_PROMPT_VARIANTS=(
  "${VACE_PROMPT_LAYOUT_PREFIX}Render a realistic sterile operating room background with clean walls, ceiling lights, surgical lamp positions, monitor stands, cabinets, IV poles, drapes, floor material, reflections, background equipment behind the workspace, moderate clutter, and neutral clinical illumination. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Inpaint the masked background as a modern hospital operating room with matte white walls, stainless storage carts, ceiling-mounted surgical lights, distant monitors, blue sterile drapes, polished floor reflections, and soft cool overhead lighting. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Generate a realistic clinical background with wall-mounted equipment rails, anesthesia monitors in the distance, cable bundles routed along the wall, stainless cabinets, covered instrument tables, and clean tiled flooring. Match the existing depth, lens perspective, lighting direction, and reflections. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Replace only the masked background with a photoreal operating room: bright surgical lamps, ceiling booms, sterile blue covers, wall consoles, equipment carts, distant display screens, smooth vinyl flooring, and subtle specular reflections. Keep the environment plausible and aligned across the views without adding foreground objects. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Inpaint a clean operating suite background with warm-white surgical lights, stainless steel fixtures, covered trays in the distance, wall cabinets, medical gas outlets, monitor arms, and soft shadows on the floor. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Fill the masked background with a sterile OR scene containing pale walls, ceiling light panels, a surgical light boom, distant instrument trolleys, monitor stands, organized cables, draped equipment, and cool balanced illumination. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Inpaint only the background with a high-fidelity surgical room: clean wall panels, overhead lamps, equipment columns, medical displays, stainless carts, blue sterile cloth, subtle floor reflections, and realistic shadowing. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Render a realistic factory background with high ceilings, exposed steel beams, overhead industrial light panels, concrete flooring, yellow safety lines, metal workbenches, tool cabinets, and distant machinery behind the workspace. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Inpaint the masked background as a clean industrial factory floor with painted concrete, safety railings, wall-mounted utility boxes, cable trays, storage carts, palletized equipment, and cool neutral overhead lighting. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Generate a photoreal manufacturing-cell background with metal shelving, assembly benches, machine enclosures, warning stripes on the floor, distant control panels, and soft reflections on polished concrete. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Replace only the masked background with an organized factory workspace: gray wall panels, suspended lights, industrial racks, tool chests, carts, floor markings, and background equipment aligned to the camera perspective. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Render a realistic warehouse background with tall pallet racks, stacked cardboard boxes, concrete floors, aisle markings, loading-bay doors in the distance, safety posts, and broad overhead LED lighting. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Inpaint the masked background as a modern logistics warehouse with metal storage shelves, wrapped pallets, plastic bins, distant dock equipment, gray concrete walls, and diffuse industrial illumination. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Generate a high-fidelity warehouse aisle background with racking columns, inventory boxes, floor scuff marks, safety bollards, ceiling trusses, and realistic shadows falling across the concrete floor. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Fill the masked background with an orderly warehouse work area: pallet shelves, storage totes, distant rolling carts, painted floor lanes, wall-mounted panels, and neutral white industrial lighting. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Render a dim sterile operating room background with low ambient ceiling lights, a few muted surgical lamps, shadowed stainless carts, dark monitor screens, pale wall panels, subtle floor reflections, and cool blue-gray clinical illumination. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Inpaint the masked background as a low-light hospital operating suite with reduced overhead lighting, softly glowing medical displays, shadowed cabinets, draped equipment in the distance, and subdued reflections on the vinyl floor. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Generate a realistic dark clinical background with surgical booms and monitor stands mostly in shadow, narrow pools of cool task light, stainless fixtures catching small highlights, and dim wall-mounted equipment rails. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Replace only the masked background with a night-shift operating room: low ceiling panel brightness, quiet dark corners, distant equipment silhouettes, faint blue sterile drapes, and controlled surgical task lighting. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Render a realistic dim factory background with high ceilings, partially lit steel beams, shadowed concrete floor, darkened machine enclosures, faint yellow safety lines, and sparse cool overhead industrial lights. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Inpaint the masked background as a low-light industrial factory cell with muted wall panels, tool cabinets in shadow, distant control panels glowing softly, cable trays overhead, and subdued reflections on polished concrete. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Generate a dark manufacturing-floor background with limited task lighting, metal shelving fading into shadow, machine guards, floor warning stripes, and realistic high-contrast industrial reflections. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Render a dim warehouse background with tall pallet racks receding into shadow, stacked boxes, dark concrete aisles, soft pools of overhead LED light, safety posts, and faint loading-bay shapes in the distance. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Inpaint the masked background as a nighttime logistics warehouse with low ambient light, muted gray walls, shadowed storage shelves, wrapped pallets, small indicator lights, and subtle floor scuff reflections. ${BK_INSTRUCTION_PROMPT}"
  "${VACE_PROMPT_LAYOUT_PREFIX}Generate a high-fidelity low-light warehouse aisle with racking columns, inventory boxes, safety bollards, ceiling trusses fading into darkness, and realistic shadows across the concrete floor. ${BK_INSTRUCTION_PROMPT}"
)

if [[ "${VACE_PROMPT_VARIANT}" == "list" ]]; then
  for i in "${!DEFAULT_BACKGROUND_INPAINT_PROMPT_VARIANTS[@]}"; do
    printf '[%02d] %s\n' "$i" "${DEFAULT_BACKGROUND_INPAINT_PROMPT_VARIANTS[$i]}"
  done
  exit 0
fi

if [[ -z "${VACE_PROMPT:-}" ]]; then
  if ! [[ "$VACE_PROMPT_VARIANT" =~ ^[0-9]+$ ]]; then
    echo "VACE_PROMPT_VARIANT must be an integer index or 'list'; got '$VACE_PROMPT_VARIANT'." >&2
    exit 2
  fi
  if (( VACE_PROMPT_VARIANT < 0 || VACE_PROMPT_VARIANT >= ${#DEFAULT_BACKGROUND_INPAINT_PROMPT_VARIANTS[@]} )); then
    echo "VACE_PROMPT_VARIANT must be in [0, $((${#DEFAULT_BACKGROUND_INPAINT_PROMPT_VARIANTS[@]} - 1))]; got '$VACE_PROMPT_VARIANT'." >&2
    exit 2
  fi
  VACE_PROMPT="${DEFAULT_BACKGROUND_INPAINT_PROMPT_VARIANTS[$VACE_PROMPT_VARIANT]}"
fi

ASSEMBLE_DIR="$ISAACLAB_ROOT/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar"
ISAAC_OUT="$OUT_DIR/isaac"
VACE_OUT="$OUT_DIR/vace"
SIDE_BY_SIDE="$OUT_DIR/vla_raw_vs_vace.mp4"
SCOPE_PY="${SCOPE_PY:-$SCOPE_ROOT/.venv/bin/python}"
if [[ ! -x "$SCOPE_PY" ]]; then
  SCOPE_PY="python"
fi

mkdir -p "$OUT_DIR"

set +u
source /localhome/local-pengfeig/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab_develop_6.0_pg
set -u

if [[ -e /lib/aarch64-linux-gnu/libgomp.so.1 ]]; then
  export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export DAYDREAM_SCOPE_MODELS_DIR="$SCOPE_MODELS_DIR"

cd "$ISAACLAB_ROOT"
BLUE_TABLE_MASK_ARGS=()
case "$MASK_PRESERVE_BLUE_TABLE" in
  1|true|TRUE|yes|YES)
    BLUE_TABLE_MASK_ARGS=(
      --mask_preserve_blue_table
      --mask_blue_table_min_blue "$MASK_BLUE_TABLE_MIN_BLUE"
      --mask_blue_table_min_green "$MASK_BLUE_TABLE_MIN_GREEN"
      --mask_blue_table_max_red "$MASK_BLUE_TABLE_MAX_RED"
      --mask_blue_table_min_blue_minus_red "$MASK_BLUE_TABLE_MIN_BLUE_MINUS_RED"
      --mask_blue_table_min_green_minus_red "$MASK_BLUE_TABLE_MIN_GREEN_MINUS_RED"
      --mask_blue_table_min_x_frac "$MASK_BLUE_TABLE_MIN_X_FRAC"
      --mask_blue_table_min_y_frac "$MASK_BLUE_TABLE_MIN_Y_FRAC"
    )
    ;;
  0|false|FALSE|no|NO)
    ;;
  *)
    echo "MASK_PRESERVE_BLUE_TABLE must be 1/0, true/false, or yes/no; got '$MASK_PRESERVE_BLUE_TABLE'." >&2
    exit 2
    ;;
esac

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
  --mask_preserve_classes "$MASK_PRESERVE_CLASSES" \
  --mask_dilate_px "$MASK_DILATE_PX" \
  "${BLUE_TABLE_MASK_ARGS[@]}" \
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
VACE_DEPTH_ARGS=()
case "$USE_VACE_DEPTH" in
  1|true|TRUE|yes|YES)
    VACE_DEPTH_ARGS=(--depth_video "$ISAAC_OUT/front_camera_depth_inverse.mp4")
    ;;
  0|false|FALSE|no|NO)
    ;;
  *)
    echo "USE_VACE_DEPTH must be 1/0, true/false, or yes/no; got '$USE_VACE_DEPTH'." >&2
    exit 2
    ;;
esac
VACE_REF_ARGS=()
if [[ -n "$VACE_REF_IMAGES" ]]; then
  VACE_REF_ARGS=(--ref_images "$VACE_REF_IMAGES" --ref_images_every_chunk "$VACE_REF_IMAGES_EVERY_CHUNK")
fi

"$SCOPE_PY" -m scope.core.pipelines.longlive.isaac_vace_background_inpaint_demo \
  --input_video "$ISAAC_OUT/front_camera_rgb.mp4" \
  --mask_video "$ISAAC_OUT/front_camera_background_mask.mp4" \
  "${VACE_DEPTH_ARGS[@]}" \
  "${VACE_REF_ARGS[@]}" \
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
  --history_conditioning "$VACE_HISTORY_CONDITIONING" \
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

case "$VACE_PRESENTATION_OUTPUT" in
  raw)
    VACE_PRESENTATION_VIDEO="$VACE_OUT/vace_background_inpaint_raw.mp4"
    ;;
  composite)
    VACE_PRESENTATION_VIDEO="$VACE_OUT/vace_background_inpaint.mp4"
    ;;
  *)
    echo "VACE_PRESENTATION_OUTPUT must be 'raw' or 'composite', got '$VACE_PRESENTATION_OUTPUT'." >&2
    exit 2
    ;;
esac

"$SCOPE_PY" - <<PY
from pathlib import Path
import cv2
import imageio.v2 as imageio
import numpy as np

left_path = Path("$ISAAC_OUT/front_camera_rgb.mp4")
right_path = Path("$VACE_PRESENTATION_VIDEO")
side_path = Path("$SIDE_BY_SIDE")
frames_dir = side_path.parent / "frames"
frames_dir.mkdir(parents=True, exist_ok=True)
left_reader = imageio.get_reader(str(left_path))
right_reader = imageio.get_reader(str(right_path))
writer = imageio.get_writer(str(side_path), fps=$FPS, macro_block_size=1)
snap_indices = {0, 6, 12, 24, 36, 44, 47}
count = 0
try:
    for idx, (left, right) in enumerate(zip(left_reader, right_reader)):
        left = cv2.resize(left[..., :3], ($VACE_WIDTH, $VACE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(right[..., :3], ($VACE_WIDTH, $VACE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        frame = np.concatenate([left, right], axis=1).astype(np.uint8)
        writer.append_data(frame)
        if idx in snap_indices:
            imageio.imwrite(str(frames_dir / f"frame{idx:03d}.png"), frame)
        count += 1
finally:
    writer.close()
    left_reader.close()
    right_reader.close()
print(f"Wrote side-by-side presentation video with {count} frames: {side_path}")
PY

python - <<PY
from pathlib import Path
print("Demo outputs:")
for path in [
    Path("$ISAAC_OUT/front_camera_rgb.mp4"),
    Path("$ISAAC_OUT/front_camera_depth_inverse.mp4"),
    Path("$ISAAC_OUT/front_camera_background_mask.mp4"),
    Path("$VACE_OUT/vace_background_inpaint.mp4"),
    Path("$VACE_OUT/vace_background_inpaint_raw.mp4"),
    Path("$VACE_PRESENTATION_VIDEO"),
    Path("$SIDE_BY_SIDE"),
]:
    print(f"  {path}")
PY
