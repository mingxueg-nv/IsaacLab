# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Save RGB, depth, and segmentation images for the assemble_trocar task.

Runs the multimodal variant of the env (rgb + depth + semantic + instance seg)
and dumps a set of PNGs per camera per env into an output directory so the
observations can be visually inspected.

Usage::

    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/save_camera_observations.py \
        --num_envs 1 --num_frames 5 --output_dir /tmp/assemble_trocar_obs --headless --enable_cameras
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Save RGB/depth/segmentation observations from the assemble_trocar env.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs to spawn.")
parser.add_argument("--num_frames", type=int, default=5, help="Number of rendered frames to save per env.")
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=3,
    help="Simulation steps to run before capturing frames (gives cameras time to produce valid output).",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="/tmp/assemble_trocar_obs",
    help="Directory to write PNG observations to.",
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Assemble-Trocar-G129-Dex3-RLinf-MultiModal-v0",
    help="Gym task id. Must be a multimodal variant that exposes depth/seg.",
)
parser.add_argument(
    "--fill_classes",
    type=str,
    nargs="*",
    default=["trocar", "trocar_device"],
    help=(
        "Semantic class names whose masks should have OmniGlass shaft-fill applied. "
        "Pass an empty list ``--fill_classes`` to disable post-processing entirely."
    ),
)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Cameras must be enabled for rendering; force it on regardless of user flag.
args_cli.enable_cameras = True

print(f"[save_camera_observations] parsed args: {args_cli}", flush=True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("[save_camera_observations] AppLauncher up", flush=True)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

# Reuse the OmniGlass shaft-fill helpers from scripts/tools/fill_trocar_mask.py so the
# live-saved masks get the same post-processing as exported datasets.
_REPO_ROOT = Path(__file__).resolve().parents[6]
_TOOLS_DIR = _REPO_ROOT / "scripts" / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
from fill_trocar_mask import _fill_frame  # noqa: E402

# View-consistent class-name → RGB palette. Every camera has its own integer
# label allocation (``idToLabels`` table is per-render-product), so we colorize
# through the class name rather than the raw ID to keep colors stable across
# the front and wrist views.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "BACKGROUND": (0, 0, 0),          # annotator miss / sky dome — black
    "UNLABELLED": (120, 120, 120),    # Kit default — gray
    "robot":              (0, 200, 0),      # green
    "trocar":             (255, 0, 255),    # magenta
    "trocar_device":      (255, 80, 80),    # red/pink
    "tray":               (80, 80, 255),    # blue
    "cart":               (255, 255, 0),    # yellow
    "ground":             (0, 255, 255),    # cyan
    "instrument_trolley": (255, 140, 0),    # orange
}
UNKNOWN_CLASS_COLOR: tuple[int, int, int] = (200, 200, 200)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().contiguous().numpy()


def _save_rgb(image: np.ndarray, path: Path) -> None:
    """Save an RGB uint8 array as PNG. Accepts (H, W, 3) or (H, W, 4)."""
    from PIL import Image

    arr = image
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        Image.fromarray(arr, mode="RGBA").save(path)
    else:
        Image.fromarray(arr[..., :3], mode="RGB").save(path)


def _save_depth(depth: np.ndarray, path: Path) -> None:
    """Save metric depth as a Cosmos-style inverse-depth PNG (near=bright, far=dark).

    Also writes the raw metric depth as ``*.npy`` for downstream use. The PNG matches
    Cosmos-Transfer2.5's convention: per-frame min-max normalization of inverse depth
    (disparity-like) into ``[0, 255]`` uint8. See
    ``cosmos_transfer2/_src/transfer2/inference/utils.py:565``.
    """
    from PIL import Image

    arr = depth
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    # Persist raw metric depth for downstream consumers.
    raw_path = path.with_suffix(".npy")
    np.save(raw_path, arr)

    valid = np.isfinite(arr) & (arr > 0)
    if not valid.any():
        Image.fromarray(np.zeros_like(arr, dtype=np.uint8), mode="L").save(path)
        return

    inv = np.zeros_like(arr, dtype=np.float32)
    inv[valid] = 1.0 / arr[valid]

    vmin = float(inv[valid].min())
    vmax = float(inv[valid].max())
    span = max(vmax - vmin, 1e-8)
    inv_norm = np.where(valid, (inv - vmin) / span, 0.0)
    png = (np.clip(inv_norm, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(png, mode="L").save(path)


def _build_id_color_lut(env, camera_name: str) -> np.ndarray:
    """Return an ``(N, 3)`` uint8 LUT mapping integer label IDs to stable RGB colors.

    Colors are looked up by class *name* in :data:`CLASS_COLORS` so the same class
    renders with the same color in every camera view, even though each camera
    allocates its own numeric IDs.
    """
    sensor = env.unwrapped.scene.sensors[camera_name]
    info = getattr(sensor.data, "info", {}) or {}
    meta = info.get("semantic_segmentation") or {}
    id_to_labels = meta.get("idToLabels") or {}

    max_id = 0
    parsed: dict[int, list[str]] = {}
    for key, label in id_to_labels.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        max_id = max(max_id, idx)
        if isinstance(label, dict):
            names = [str(v) for v in label.values()]
        else:
            names = [str(label)]
        parsed[idx] = names

    lut = np.full((max_id + 1, 3), UNKNOWN_CLASS_COLOR, dtype=np.uint8)
    for idx, names in parsed.items():
        color = UNKNOWN_CLASS_COLOR
        for n in names:
            if n in CLASS_COLORS:
                color = CLASS_COLORS[n]
                break
        lut[idx] = color
    return lut


def _save_segmentation(
    seg: np.ndarray,
    path: Path,
    fill_label_ids: list[int],
    id_color_lut: np.ndarray,
) -> None:
    """Save a semantic-segmentation frame, closing transparent trocar shafts.

    The multimodal env configures semantic segmentation as a non-colorized int32
    class-id map. For the label IDs in ``fill_label_ids`` (resolved from class
    names via the camera's ``idToLabels`` mapping), we run
    :func:`fill_trocar_mask._fill_frame` so the glass shaft (invisible under
    OmniGlass) is bridged between the tip and the handle. Every other label is
    left untouched — applying the fill to labels like ``background`` or ``robot``
    produces spurious rectangles between unrelated regions.

    - ``<path>.npy``  — post-processed int32 label map (H, W)
    - ``<path>``      — colorized PNG using :data:`CLASS_COLORS` via
      ``id_color_lut`` so the same class keeps the same color across cameras.

    Colorized (RGBA uint8) inputs are saved as-is and not post-processed, since
    the fill algorithm operates on integer labels.
    """
    from PIL import Image

    arr = seg
    if arr.ndim == 3 and arr.shape[-1] == 4 and arr.dtype == np.uint8:
        Image.fromarray(arr, mode="RGBA").save(path)
        return

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    arr = arr.astype(np.int32)

    present = set(int(x) for x in np.unique(arr))
    active = [lid for lid in fill_label_ids if lid in present]
    if active:
        arr = _fill_frame(arr, active)

    raw_path = path.with_suffix(".npy")
    np.save(raw_path, arr)

    H, W = arr.shape
    rgb = np.full((H, W, 3), UNKNOWN_CLASS_COLOR, dtype=np.uint8)
    if id_color_lut.shape[0] > 0:
        in_range = (arr >= 0) & (arr < id_color_lut.shape[0])
        rgb[in_range] = id_color_lut[arr[in_range]]
    Image.fromarray(rgb, mode="RGB").save(path)


def _resolve_fill_label_ids(env, camera_names: list[str], class_names: list[str]) -> dict[str, list[int]]:
    """Look up the integer label IDs corresponding to ``class_names`` for each camera.

    Returns a ``{camera_name: [label_id, ...]}`` mapping. Each camera has its own
    ``idToLabels`` dict and the IDs are not guaranteed to match across cameras.
    """
    if not class_names:
        return {cam: [] for cam in camera_names}
    wanted = {c.lower() for c in class_names}
    out: dict[str, list[int]] = {}
    for cam in camera_names:
        sensor = env.unwrapped.scene.sensors[cam]
        info = getattr(sensor.data, "info", {}) or {}
        meta = info.get("semantic_segmentation") or {}
        id_to_labels = meta.get("idToLabels") or {}
        ids: list[int] = []
        for key, label in id_to_labels.items():
            # ``label`` is either a dict like ``{"class": "trocar"}`` or a string.
            if isinstance(label, dict):
                values = {str(v).lower() for v in label.values()}
            else:
                values = {str(label).lower()}
            if values & wanted:
                try:
                    ids.append(int(key))
                except (TypeError, ValueError):
                    continue
        out[cam] = sorted(set(ids))
    return out


def _iter_camera_obs(obs_group: dict[str, torch.Tensor]):
    """Yield (camera_name, tensor) for each camera key in a camera obs group."""
    for key, value in obs_group.items():
        if isinstance(value, torch.Tensor):
            yield key, value


def _compute_hold_action(env) -> torch.Tensor:
    """Build an action tensor that commands the robot to hold its current pose.

    The joint_pos action term applies ``targets = scale * action + offset``, so to
    hold pose we set ``action = (current_joint_pos - offset) / scale``.
    """
    import warp as wp

    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    robot = env.unwrapped.scene["robot"]
    joint_pos = robot.data.joint_pos
    if not isinstance(joint_pos, torch.Tensor):
        joint_pos = wp.to_torch(joint_pos)

    # The action term stores the joint indices it drives, in action order.
    joint_ids = action_term._joint_ids
    if isinstance(joint_ids, slice):
        selected = joint_pos[:, joint_ids]
    else:
        ids_tensor = torch.as_tensor(list(joint_ids), dtype=torch.long, device=joint_pos.device)
        selected = joint_pos.index_select(dim=1, index=ids_tensor)

    # Subtract offset and divide by scale to recover pre-action values.
    offset = action_term._offset
    scale = action_term._scale
    if isinstance(offset, torch.Tensor):
        offset = offset.to(selected.device)
    if isinstance(scale, torch.Tensor):
        scale = scale.to(selected.device)
    action = (selected - offset) / scale
    return action.contiguous()


def main() -> None:
    print("[save_camera_observations] entered main()", flush=True)
    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=getattr(args_cli, "device", None) or "cuda:0",
        num_envs=args_cli.num_envs,
    )
    print("[save_camera_observations] env_cfg parsed", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("[save_camera_observations] gym.make complete", flush=True)

    try:
        print(f"[INFO] Task: {args_cli.task}", flush=True)
        print(f"[INFO] Num envs: {env.unwrapped.num_envs}", flush=True)
        print(f"[INFO] Output directory: {out_dir.resolve()}", flush=True)

        obs, _ = env.reset()
        print(f"[INFO] env.reset done; obs keys: {list(obs.keys())}", flush=True)

        hold_action = _compute_hold_action(env)
        print(f"[INFO] hold_action shape: {tuple(hold_action.shape)}", flush=True)

        # Warmup: cameras usually need a few ticks before producing valid images.
        for i in range(max(args_cli.warmup_steps, 0)):
            obs, *_ = env.step(hold_action)
            print(f"[INFO] warmup step {i + 1}/{args_cli.warmup_steps}", flush=True)

        camera_names = list(obs.get("camera_semantic_segmentation", {}).keys())
        fill_ids_per_cam = _resolve_fill_label_ids(env, camera_names, args_cli.fill_classes)
        print(
            f"[INFO] fill_classes={args_cli.fill_classes}  "
            f"resolved_ids_per_camera={fill_ids_per_cam}",
            flush=True,
        )

        for frame_idx in range(args_cli.num_frames):
            obs, *_ = env.step(hold_action)

            rgb_group: dict[str, torch.Tensor] = obs.get("camera_images", {})
            depth_group: dict[str, torch.Tensor] = obs.get("camera_depth", {})
            semantic_group: dict[str, torch.Tensor] = obs.get("camera_semantic_segmentation", {})

            for env_idx in range(env.unwrapped.num_envs):
                env_dir = out_dir / f"env_{env_idx:02d}" / f"frame_{frame_idx:03d}"
                env_dir.mkdir(parents=True, exist_ok=True)

                for cam_name, tensor in _iter_camera_obs(rgb_group):
                    arr = _to_numpy(tensor[env_idx])
                    _save_rgb(arr, env_dir / f"{cam_name}_rgb.png")

                for cam_name, tensor in _iter_camera_obs(depth_group):
                    arr = _to_numpy(tensor[env_idx])
                    _save_depth(arr, env_dir / f"{cam_name}_depth.png")

                for cam_name, tensor in _iter_camera_obs(semantic_group):
                    arr = _to_numpy(tensor[env_idx])
                    # Rebuild LUT each frame because the annotator's idToLabels
                    # can grow as new classes first come into view.
                    _save_segmentation(
                        arr,
                        env_dir / f"{cam_name}_semantic.png",
                        fill_label_ids=fill_ids_per_cam.get(cam_name, []),
                        id_color_lut=_build_id_color_lut(env, cam_name),
                    )

            print(f"[INFO] Saved frame {frame_idx + 1}/{args_cli.num_frames}", flush=True)

        print(f"[DONE] Images written to {out_dir.resolve()}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - diagnostic
        import traceback

        print(f"[ERROR] {exc!r}", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
        sys.exit(0)
