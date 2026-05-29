from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from contextlib import ExitStack
from pathlib import Path

parser = argparse.ArgumentParser(description="Run GR00T policy chunks in Assemble Trocar and capture RGB/depth.")
parser.add_argument("--model_path", required=True)
parser.add_argument("--config_dir", required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_frames", type=int, default=96)
parser.add_argument("--warmup_steps", type=int, default=4)
parser.add_argument("--output_dir", type=str, default="/tmp/gr00t_isaac_demo_stream")
parser.add_argument("--camera", type=str, default="front_camera")
parser.add_argument("--camera_width", type=int, default=640)
parser.add_argument("--camera_height", type=int, default=480)
parser.add_argument("--policy_width", type=int, default=224)
parser.add_argument("--policy_height", type=int, default=224)
parser.add_argument("--fps", type=int, default=12)
parser.add_argument("--denoising_steps", type=int, default=4)
parser.add_argument("--task_description", default="install trocar from box")
parser.add_argument(
    "--background_mask",
    action="store_true",
    help="Write a white=background/inpaint, black=foreground/preserve mask for the selected camera.",
)
parser.add_argument(
    "--mask_preserve_classes",
    default="robot,tray,cart,trocar,trocar_device",
    help="Comma-separated semantic classes to preserve as foreground when --background_mask is enabled.",
)
parser.add_argument(
    "--mask_dilate_px",
    type=int,
    default=0,
    help="Dilate the preserved foreground mask by this many pixels to protect object boundaries.",
)
parser.add_argument(
    "--mask_preserve_blue_table",
    action="store_true",
    help="Preserve the lower-right solid blue tabletop as foreground using a targeted RGB+ROI mask.",
)
parser.add_argument("--mask_blue_table_min_blue", type=int, default=95)
parser.add_argument("--mask_blue_table_min_green", type=int, default=70)
parser.add_argument("--mask_blue_table_max_red", type=int, default=135)
parser.add_argument("--mask_blue_table_min_blue_minus_red", type=int, default=25)
parser.add_argument("--mask_blue_table_min_green_minus_red", type=int, default=5)
parser.add_argument("--mask_blue_table_min_x_frac", type=float, default=0.48)
parser.add_argument("--mask_blue_table_min_y_frac", type=float, default=0.30)
parser.add_argument(
    "--render_antialiasing",
    choices=("inherit", "Off", "FXAA", "DLSS", "TAA", "DLAA"),
    default="DLAA",
)
parser.add_argument(
    "--render_translucency",
    choices=("inherit", "on", "off"),
    default="off",
)
parser.add_argument(
    "--wrist_camera_update_period",
    type=float,
    default=0.0,
    help=(
        "When > 0, update left/right wrist RGB cameras at this period in seconds. "
        "The front camera remains frame-rate because it drives the live view, depth, and mask."
    ),
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Assemble-Trocar-G129-Dex3-RLinf-MultiModal-v0",
)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

_TASK_DIR = Path(__file__).resolve().parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))
from _obs_helpers import depth_to_inverse_rgb, resolve_fill_label_ids, to_numpy  # noqa: E402

config_dir = Path(args_cli.config_dir).resolve()
if str(config_dir) not in sys.path:
    sys.path.insert(0, str(config_dir))
from gr00t.model.policy import Gr00tPolicy  # noqa: E402
from gr00t_config import IsaacLabDataConfig  # noqa: E402

CAMERA_NAMES = ("front_camera", "left_wrist_camera", "right_wrist_camera")
ACTION_KEYS = ("action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand")


def _uint8_rgb(frame: np.ndarray) -> np.ndarray:
    arr = frame
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


def _resize_for_policy(frame: np.ndarray) -> np.ndarray:
    rgb = _uint8_rgb(frame)
    image = Image.fromarray(rgb)
    image = image.resize((args_cli.policy_width, args_cli.policy_height), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}
    p95_index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return {
        "count": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p95_ms": ordered[p95_index],
    }


def _trim_env_cfg_to_policy_sensors(env_cfg, depth_camera_name: str, include_semantic: bool) -> None:
    if depth_camera_name not in CAMERA_NAMES:
        raise ValueError(f"Unknown camera {depth_camera_name!r}. Expected one of {CAMERA_NAMES}.")

    for name in CAMERA_NAMES:
        camera_cfg = getattr(env_cfg.scene, name, None)
        if camera_cfg is None:
            continue
        camera_cfg.width = args_cli.camera_width
        camera_cfg.height = args_cli.camera_height
        if name in ("left_wrist_camera", "right_wrist_camera") and args_cli.wrist_camera_update_period > 0.0:
            camera_cfg.update_period = args_cli.wrist_camera_update_period
        if name == depth_camera_name:
            camera_cfg.data_types = ["rgb", "distance_to_image_plane"]
            if include_semantic:
                camera_cfg.data_types.append("semantic_segmentation")
                camera_cfg.colorize_semantic_segmentation = False
        else:
            camera_cfg.data_types = ["rgb"]

    observations = getattr(env_cfg, "observations", None)
    if observations is None:
        return

    depth_group = getattr(observations, "camera_depth", None)
    if depth_group is not None:
        for name in CAMERA_NAMES:
            if name != depth_camera_name and hasattr(depth_group, name):
                setattr(depth_group, name, None)

    if hasattr(observations, "camera_semantic_segmentation") and not include_semantic:
        observations.camera_semantic_segmentation = None
    elif hasattr(observations, "camera_semantic_segmentation"):
        semantic_group = observations.camera_semantic_segmentation
        for name in CAMERA_NAMES:
            if name != depth_camera_name and hasattr(semantic_group, name):
                setattr(semantic_group, name, None)


def _apply_render_overrides(env_cfg) -> None:
    if args_cli.render_antialiasing != "inherit":
        env_cfg.sim.render.antialiasing_mode = args_cli.render_antialiasing
    if args_cli.render_translucency != "inherit":
        env_cfg.sim.render.enable_translucency = args_cli.render_translucency == "on"
        if args_cli.render_translucency == "off":
            env_cfg.sim.render.carb_settings = None


def _build_policy_obs(obs: dict) -> dict[str, object]:
    rgb_group = obs["camera_images"]
    policy_group = obs["policy"]

    room = _resize_for_policy(to_numpy(rgb_group["front_camera"][0]))
    left = _resize_for_policy(to_numpy(rgb_group["left_wrist_camera"][0]))
    right = _resize_for_policy(to_numpy(rgb_group["right_wrist_camera"][0]))

    body = to_numpy(policy_group["robot_joint_state"])[0, 15:29].astype(np.float32)
    hands = to_numpy(policy_group["robot_dex3_joint_state"])[0].astype(np.float32)
    state = np.concatenate([body, hands], axis=0)

    return {
        "video.left_wrist_view": left[None],
        "video.right_wrist_view": right[None],
        "video.room_view": room[None],
        "state.left_arm": state[0:7][None],
        "state.right_arm": state[7:14][None],
        "state.left_hand": state[14:21][None],
        "state.right_hand": state[21:28][None],
        "annotation.human.task_description": np.array([args_cli.task_description]),
    }


def _action_chunk_to_isaac(action_chunk: dict[str, np.ndarray]) -> np.ndarray:
    parts = [np.asarray(action_chunk[key], dtype=np.float32) for key in ACTION_KEYS]
    action = np.concatenate(parts, axis=-1)
    if action.shape[-1] != 28:
        raise ValueError(f"Expected 28 GR00T action dims before padding, got {action.shape}.")
    return np.pad(action, ((0, 0), (15, 0)), mode="constant", constant_values=0.0)


def _parse_classes(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _as_label_map(seg: np.ndarray) -> np.ndarray:
    arr = seg
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[-1] == 4 and arr.dtype == np.uint8:
        raise ValueError("Expected non-colorized semantic_segmentation int labels, got colorized RGBA.")
    return arr.astype(np.int32)


def _dilate_binary(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    size = max(3, pixels | 1)
    kernel = np.ones((size, size), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def _background_mask_from_semantic(seg: np.ndarray, preserve_ids: list[int], dilate_px: int) -> np.ndarray:
    label_map = _as_label_map(seg)
    foreground = np.isin(label_map, np.asarray(preserve_ids, dtype=np.int32))
    foreground = _dilate_binary(foreground, dilate_px)
    return ((~foreground).astype(np.uint8) * 255)


def _blue_table_foreground_mask(
    rgb: np.ndarray,
    min_blue: int,
    min_green: int,
    max_red: int,
    min_blue_minus_red: int,
    min_green_minus_red: int,
    min_x_frac: float,
    min_y_frac: float,
) -> np.ndarray:
    """Target the solid blue tabletop without preserving the upper grid floor."""
    arr = rgb.astype(np.int16)
    red = arr[..., 0]
    green = arr[..., 1]
    blue = arr[..., 2]
    height, width = rgb.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    blue_pixels = (
        (blue >= min_blue)
        & (green >= min_green)
        & (red <= max_red)
        & ((blue - red) >= min_blue_minus_red)
        & ((green - red) >= min_green_minus_red)
    )
    lower_right_roi = (xx >= int(width * min_x_frac)) & (yy >= int(height * min_y_frac))
    return blue_pixels & lower_right_roi


def main() -> None:
    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=getattr(args_cli, "device", None) or "cuda:0",
        num_envs=args_cli.num_envs,
    )
    _trim_env_cfg_to_policy_sensors(env_cfg, args_cli.camera, args_cli.background_mask)
    _apply_render_overrides(env_cfg)

    data_config = IsaacLabDataConfig()
    load_t0 = time.perf_counter()
    policy = Gr00tPolicy(
        model_path=args_cli.model_path,
        embodiment_tag="new_embodiment",
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        denoising_steps=args_cli.denoising_steps,
        device="cuda:0",
    )
    torch.cuda.synchronize()
    policy_load_s = time.perf_counter() - load_t0

    env = gym.make(args_cli.task, cfg=env_cfg)

    rgb_path = out_dir / f"{args_cli.camera}_rgb.mp4"
    depth_path = out_dir / f"{args_cli.camera}_depth_inverse.mp4"
    mask_path = out_dir / f"{args_cli.camera}_background_mask.mp4"
    meta_path = out_dir / "metrics.json"

    policy_chunk_ms: list[float] = []
    step_ms: list[float] = []
    capture_ms: list[float] = []
    total_ms: list[float] = []
    mask_coverage: list[float] = []
    preserve_classes = _parse_classes(args_cli.mask_preserve_classes)
    preserve_ids: list[int] = []

    try:
        obs, _ = env.reset()
        zero_action = torch.zeros((args_cli.num_envs, 43), dtype=torch.float32, device="cuda:0")
        for _ in range(max(args_cli.warmup_steps, 0)):
            obs, *_ = env.step(zero_action)
        torch.cuda.synchronize()

        if args_cli.background_mask:
            resolved = resolve_fill_label_ids(env, [args_cli.camera], preserve_classes)
            preserve_ids = resolved.get(args_cli.camera, [])
            if not preserve_ids:
                raise RuntimeError(
                    f"No semantic IDs resolved for preserve classes {preserve_classes} on {args_cli.camera}."
                )
            print(
                f"[mask] preserve_classes={preserve_classes} preserve_ids={preserve_ids} "
                f"dilate_px={args_cli.mask_dilate_px}",
                flush=True,
            )

        action_chunk = None
        action_index = 0

        with ExitStack() as stack:
            rgb_writer = stack.enter_context(imageio.get_writer(str(rgb_path), fps=args_cli.fps, macro_block_size=1))
            depth_writer = stack.enter_context(imageio.get_writer(str(depth_path), fps=args_cli.fps, macro_block_size=1))
            mask_writer = None
            if args_cli.background_mask:
                mask_writer = stack.enter_context(
                    imageio.get_writer(str(mask_path), fps=args_cli.fps, macro_block_size=1)
                )

            for frame_idx in range(args_cli.num_frames):
                t_total0 = time.perf_counter()
                if action_chunk is None or action_index >= action_chunk.shape[0]:
                    policy_obs = _build_policy_obs(obs)
                    torch.cuda.synchronize()
                    t_policy0 = time.perf_counter()
                    action_chunk = _action_chunk_to_isaac(policy.get_action(policy_obs))
                    torch.cuda.synchronize()
                    policy_chunk_ms.append((time.perf_counter() - t_policy0) * 1000.0)
                    action_index = 0

                action = torch.from_numpy(action_chunk[action_index]).to(device="cuda:0").view(1, -1)
                action_index += 1

                t_step0 = time.perf_counter()
                obs, *_ = env.step(action)
                torch.cuda.synchronize()
                t_step1 = time.perf_counter()

                rgb = _uint8_rgb(to_numpy(obs["camera_images"][args_cli.camera][0]))
                depth = to_numpy(obs["camera_depth"][args_cli.camera][0])
                depth_rgb = _uint8_rgb(depth_to_inverse_rgb(depth))
                rgb_writer.append_data(rgb)
                depth_writer.append_data(depth_rgb)

                mask_rgb = None
                if args_cli.background_mask:
                    semantic_group = obs.get("camera_semantic_segmentation", {})
                    if args_cli.camera not in semantic_group:
                        raise RuntimeError(f"Missing semantic segmentation for {args_cli.camera}.")
                    seg = to_numpy(semantic_group[args_cli.camera][0])
                    mask = _background_mask_from_semantic(seg, preserve_ids, args_cli.mask_dilate_px)
                    if args_cli.mask_preserve_blue_table:
                        blue_table_fg = _blue_table_foreground_mask(
                            rgb,
                            min_blue=args_cli.mask_blue_table_min_blue,
                            min_green=args_cli.mask_blue_table_min_green,
                            max_red=args_cli.mask_blue_table_max_red,
                            min_blue_minus_red=args_cli.mask_blue_table_min_blue_minus_red,
                            min_green_minus_red=args_cli.mask_blue_table_min_green_minus_red,
                            min_x_frac=args_cli.mask_blue_table_min_x_frac,
                            min_y_frac=args_cli.mask_blue_table_min_y_frac,
                        )
                        mask[blue_table_fg] = 0
                    mask_rgb = np.repeat(mask[..., None], 3, axis=-1)
                    assert mask_writer is not None
                    mask_writer.append_data(mask_rgb)
                    mask_coverage.append(float(mask.mean() / 255.0))

                t_capture = time.perf_counter()

                step_ms.append((t_step1 - t_step0) * 1000.0)
                capture_ms.append((t_capture - t_step1) * 1000.0)
                total_ms.append((t_capture - t_total0) * 1000.0)

                if frame_idx == 0:
                    imageio.imwrite(out_dir / f"{args_cli.camera}_rgb_first.png", rgb)
                    imageio.imwrite(out_dir / f"{args_cli.camera}_depth_first.png", depth_rgb)
                    if mask_rgb is not None:
                        imageio.imwrite(out_dir / f"{args_cli.camera}_background_mask_first.png", mask_rgb)

                print(
                    f"[gr00t-isaac] frame={frame_idx + 1:04d}/{args_cli.num_frames} "
                    f"step_ms={step_ms[-1]:.2f} capture_ms={capture_ms[-1]:.2f} total_ms={total_ms[-1]:.2f} "
                    f"action_i={action_index:02d}/{action_chunk.shape[0]:02d}",
                    flush=True,
                )

        metrics = {
            "task": args_cli.task,
            "model_path": args_cli.model_path,
            "camera": args_cli.camera,
            "num_envs": args_cli.num_envs,
            "num_frames": args_cli.num_frames,
            "camera_width": args_cli.camera_width,
            "camera_height": args_cli.camera_height,
            "policy_width": args_cli.policy_width,
            "policy_height": args_cli.policy_height,
            "fps": args_cli.fps,
            "denoising_steps": args_cli.denoising_steps,
            "policy_load_s": policy_load_s,
            "rendering_mode": getattr(args_cli, "rendering_mode", None),
            "render_antialiasing": args_cli.render_antialiasing,
            "render_translucency": args_cli.render_translucency,
            "wrist_camera_update_period": args_cli.wrist_camera_update_period,
            "rgb_video": str(rgb_path),
            "depth_inverse_video": str(depth_path),
            "background_mask_video": str(mask_path) if args_cli.background_mask else None,
            "background_mask_enabled": args_cli.background_mask,
            "mask_preserve_classes": preserve_classes,
            "mask_preserve_ids": preserve_ids,
            "mask_dilate_px": args_cli.mask_dilate_px,
            "mask_preserve_blue_table": args_cli.mask_preserve_blue_table,
            "mask_blue_table_thresholds": {
                "min_blue": args_cli.mask_blue_table_min_blue,
                "min_green": args_cli.mask_blue_table_min_green,
                "max_red": args_cli.mask_blue_table_max_red,
                "min_blue_minus_red": args_cli.mask_blue_table_min_blue_minus_red,
                "min_green_minus_red": args_cli.mask_blue_table_min_green_minus_red,
                "min_x_frac": args_cli.mask_blue_table_min_x_frac,
                "min_y_frac": args_cli.mask_blue_table_min_y_frac,
            },
            "mask_coverage_mean": statistics.fmean(mask_coverage) if mask_coverage else None,
            "policy_chunk_ms": _stats(policy_chunk_ms),
            "step_ms": _stats(step_ms),
            "capture_write_ms": _stats(capture_ms),
            "total_frame_ms": _stats(total_ms),
            "raw_policy_chunk_ms": policy_chunk_ms,
            "raw_step_ms": step_ms,
            "raw_capture_write_ms": capture_ms,
            "raw_total_frame_ms": total_ms,
        }
        meta_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in metrics.items() if not k.startswith("raw_")}, indent=2), flush=True)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
