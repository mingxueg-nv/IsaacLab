from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description="Capture Assemble Trocar RGB/depth videos and render timing.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_frames", type=int, default=120)
parser.add_argument("--warmup_steps", type=int, default=8)
parser.add_argument("--output_dir", type=str, default="/tmp/assemble_trocar_demo_capture")
parser.add_argument("--camera", type=str, default="front_camera")
parser.add_argument("--camera_width", type=int, default=640)
parser.add_argument("--camera_height", type=int, default=480)
parser.add_argument("--fps", type=int, default=12)
parser.add_argument(
    "--keep_all_sensors",
    action="store_true",
    help="Keep the task's default camera/sensor set. By default only the selected RGB+depth camera is rendered.",
)
parser.add_argument(
    "--policy_sensors",
    action="store_true",
    help=(
        "Render the policy-compatible minimal sensor set: RGB from front/left/right cameras, "
        "depth only for the selected camera, and no semantic segmentation."
    ),
)
parser.add_argument(
    "--render_antialiasing",
    choices=("inherit", "Off", "FXAA", "DLSS", "TAA", "DLAA"),
    default="Off",
    help="Override IsaacLab RenderCfg.antialiasing_mode for the benchmark/demo path.",
)
parser.add_argument(
    "--render_translucency",
    choices=("inherit", "on", "off"),
    default="off",
    help="Override IsaacLab RenderCfg.enable_translucency for the benchmark/demo path.",
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
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

_TASK_DIR = Path(__file__).resolve().parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))
from _obs_helpers import compute_hold_action, depth_to_inverse_rgb, to_numpy  # noqa: E402

CAMERA_NAMES = ("front_camera", "left_wrist_camera", "right_wrist_camera")


def _uint8_rgb(frame: np.ndarray) -> np.ndarray:
    arr = frame
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


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


def _trim_env_cfg_to_camera(env_cfg, camera_name: str) -> None:
    if camera_name not in CAMERA_NAMES:
        raise ValueError(f"Unknown camera {camera_name!r}. Expected one of {CAMERA_NAMES}.")

    for name in CAMERA_NAMES:
        camera_cfg = getattr(env_cfg.scene, name, None)
        if camera_cfg is None:
            continue
        if name == camera_name:
            camera_cfg.data_types = ["rgb", "distance_to_image_plane"]
            camera_cfg.width = args_cli.camera_width
            camera_cfg.height = args_cli.camera_height
        else:
            setattr(env_cfg.scene, name, None)

    observations = getattr(env_cfg, "observations", None)
    if observations is None:
        return

    for group_name in ("camera_images", "camera_depth"):
        group = getattr(observations, group_name, None)
        if group is None:
            continue
        for name in CAMERA_NAMES:
            if name != camera_name and hasattr(group, name):
                setattr(group, name, None)

    if hasattr(observations, "camera_semantic_segmentation"):
        observations.camera_semantic_segmentation = None


def _trim_env_cfg_to_policy_sensors(env_cfg, depth_camera_name: str) -> None:
    if depth_camera_name not in CAMERA_NAMES:
        raise ValueError(f"Unknown camera {depth_camera_name!r}. Expected one of {CAMERA_NAMES}.")

    for name in CAMERA_NAMES:
        camera_cfg = getattr(env_cfg.scene, name, None)
        if camera_cfg is None:
            continue
        camera_cfg.width = args_cli.camera_width
        camera_cfg.height = args_cli.camera_height
        if name == depth_camera_name:
            camera_cfg.data_types = ["rgb", "distance_to_image_plane"]
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

    if hasattr(observations, "camera_semantic_segmentation"):
        observations.camera_semantic_segmentation = None


def _apply_render_overrides(env_cfg) -> None:
    if args_cli.render_antialiasing != "inherit":
        env_cfg.sim.render.antialiasing_mode = args_cli.render_antialiasing
    if args_cli.render_translucency != "inherit":
        env_cfg.sim.render.enable_translucency = args_cli.render_translucency == "on"
        if args_cli.render_translucency == "off":
            env_cfg.sim.render.carb_settings = None


def main() -> None:
    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=getattr(args_cli, "device", None) or "cuda:0",
        num_envs=args_cli.num_envs,
    )
    if args_cli.policy_sensors:
        _trim_env_cfg_to_policy_sensors(env_cfg, args_cli.camera)
    elif not args_cli.keep_all_sensors:
        _trim_env_cfg_to_camera(env_cfg, args_cli.camera)
    _apply_render_overrides(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)

    rgb_path = out_dir / f"{args_cli.camera}_rgb.mp4"
    depth_path = out_dir / f"{args_cli.camera}_depth_inverse.mp4"
    meta_path = out_dir / "metrics.json"

    step_ms: list[float] = []
    capture_ms: list[float] = []
    total_ms: list[float] = []

    try:
        obs, _ = env.reset()
        hold_action = compute_hold_action(env)

        for _ in range(max(args_cli.warmup_steps, 0)):
            obs, *_ = env.step(hold_action)
        torch.cuda.synchronize()

        with imageio.get_writer(str(rgb_path), fps=args_cli.fps, macro_block_size=1) as rgb_writer, imageio.get_writer(
            str(depth_path),
            fps=args_cli.fps,
            macro_block_size=1,
        ) as depth_writer:
            for frame_idx in range(args_cli.num_frames):
                t0 = time.perf_counter()
                obs, *_ = env.step(hold_action)
                torch.cuda.synchronize()
                t1 = time.perf_counter()

                rgb_group = obs.get("camera_images", {})
                depth_group = obs.get("camera_depth", {})
                if args_cli.camera not in rgb_group or args_cli.camera not in depth_group:
                    raise KeyError(f"Camera {args_cli.camera!r} not found in observation groups.")

                rgb = _uint8_rgb(to_numpy(rgb_group[args_cli.camera][0]))
                depth = to_numpy(depth_group[args_cli.camera][0])
                depth_rgb = _uint8_rgb(depth_to_inverse_rgb(depth))

                rgb_writer.append_data(rgb)
                depth_writer.append_data(depth_rgb)
                t2 = time.perf_counter()

                step_ms.append((t1 - t0) * 1000.0)
                capture_ms.append((t2 - t1) * 1000.0)
                total_ms.append((t2 - t0) * 1000.0)

                if frame_idx == 0:
                    imageio.imwrite(out_dir / f"{args_cli.camera}_rgb_first.png", rgb)
                    imageio.imwrite(out_dir / f"{args_cli.camera}_depth_first.png", depth_rgb)

                print(
                    f"[capture] frame={frame_idx + 1:04d}/{args_cli.num_frames} "
                    f"step_ms={step_ms[-1]:.2f} capture_ms={capture_ms[-1]:.2f} total_ms={total_ms[-1]:.2f}",
                    flush=True,
                )

        metrics = {
            "task": args_cli.task,
            "camera": args_cli.camera,
            "num_envs": args_cli.num_envs,
            "num_frames": args_cli.num_frames,
            "camera_width": args_cli.camera_width,
            "camera_height": args_cli.camera_height,
            "fps": args_cli.fps,
            "keep_all_sensors": args_cli.keep_all_sensors,
            "policy_sensors": args_cli.policy_sensors,
            "rendering_mode": getattr(args_cli, "rendering_mode", None),
            "render_antialiasing": args_cli.render_antialiasing,
            "render_translucency": args_cli.render_translucency,
            "rgb_video": str(rgb_path),
            "depth_inverse_video": str(depth_path),
            "step_ms": _stats(step_ms),
            "capture_write_ms": _stats(capture_ms),
            "total_ms": _stats(total_ms),
            "raw_step_ms": step_ms,
            "raw_capture_write_ms": capture_ms,
            "raw_total_ms": total_ms,
        }
        meta_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in metrics.items() if not k.startswith("raw_")}, indent=2), flush=True)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
