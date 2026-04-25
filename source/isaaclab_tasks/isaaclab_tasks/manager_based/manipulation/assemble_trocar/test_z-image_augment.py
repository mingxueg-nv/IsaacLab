# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Send sim RGB plus optional sim depth/segmentation control to Z-Image-Fun.

This mirrors ``test_cosmos_augment.py`` for the Z-Image-Fun ControlNet service:
it runs the multimodal ``assemble_trocar`` env, captures frames from each
camera, sends them over ZMQ to a running ``z_image_service.py`` process, and
writes an ``input_rgb / control / z_image_out`` set per frame.

Prereqs (the service process must already be running):

    cd Isaac-GR00T/third_party/VideoX-Fun
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python ../../scripts/z_image_service.py --port 5657

Usage::

    ./isaaclab.sh -p \\
        source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/test_z-image_augment.py \\
        --num_envs 1 --num_frames 2 --control_kind depth --z_image_port 5657 \\
        --grid_mode --output_dir outputs/z_image_augment --headless

Notes:
    * ``--control_kind edge``: sends only RGB. The Z-Image service computes a
      Canny edge control image internally from the RGB frame.
    * ``--control_kind depth``: sends a ZoeDepth-style normalized grayscale
      depth control matching VideoX-Fun's ``ImageToDepth`` convention.
    * ``--control_kind seg``: sends the same view-consistent class-name-colored
      segmentation control as ``test_cosmos_augment.py``.
    * ``--control_kind none``: disables control images entirely. This is useful
      for standalone Z-Image inpainting with ``--inpaint_background``.
    * ``--grid_mode`` tiles up to four cameras into the same 2x2 layout used by
      ``VideoZImageAugmentTransform(grid_mode=True)`` during RLinf training,
      then splits the augmented grid back into per-camera images.
    * ``--z_image_view_size HEIGHT WIDTH`` resizes each camera view before
      sending to Z-Image. In grid mode, the 2x2 grid is built from these
      resized views.
    * ``--inpaint_background`` sends RGB as the inpaint source image and a
      semantic mask where the configured foreground classes are preserved while
      the background is regenerated. ``--inpaint_preserve_dilation_px`` expands
      that preserved foreground before inpainting to protect object boundaries.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

TRAILER_MAGIC = b"CXG1"
TRAILER_TAG_CONTROL = b"CTRL"
TRAILER_TAG_INPAINT_MASK = b"IMSK"
TRAILER_TAG_NO_CONTROL = b"NOCT"

parser = argparse.ArgumentParser(description="Send sim frames to Z-Image-Fun and save augmented output.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs to spawn.")
parser.add_argument("--num_frames", type=int, default=2, help="Number of rendered frames to augment per env.")
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=3,
    help="Simulation steps to run before capturing frames.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="outputs/z_image_augment",
    help="Directory to write augmented PNGs to.",
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Assemble-Trocar-G129-Dex3-RLinf-MultiModal-v0",
    help="Gym task id. Must expose RGB camera images, and depth/seg for those control modes.",
)
parser.add_argument(
    "--fill_classes",
    type=str,
    nargs="*",
    default=["trocar", "trocar_device"],
    help="Semantic class names whose masks should get OmniGlass shaft-fill applied.",
)
parser.add_argument(
    "--control_kind",
    type=str,
    default="edge",
    choices=["edge", "depth", "seg", "none"],
    help=(
        "Which control signal to use. ``edge`` sends RGB only and lets the "
        "Z-Image service compute Canny control internally; ``depth`` and "
        "``seg`` send sim control images in the request trailer; ``none`` "
        "disables control images entirely."
    ),
)
parser.add_argument("--z_image_host", type=str, default="localhost", help="Z-Image service ZMQ host.")
parser.add_argument("--z_image_port", type=int, default=5657, help="Z-Image service ZMQ port.")
parser.add_argument(
    "--z_image_seed",
    type=int,
    default=0,
    help="Seed for Z-Image generation.",
)
parser.add_argument(
    "--z_image_timeout_s",
    type=int,
    default=300,
    help="Per-request timeout for the Z-Image service (seconds).",
)
parser.add_argument(
    "--z_image_view_size",
    type=int,
    nargs=2,
    metavar=("HEIGHT", "WIDTH"),
    default=(224, 224),
    help=(
        "Optional per-camera view size sent to Z-Image. RGB, control, and "
        "inpaint masks are resized before grid construction or per-camera "
        "requests. Default sends the original camera resolution."
    ),
)
parser.add_argument(
    "--cameras",
    type=str,
    nargs="*",
    default=["front_camera", "left_wrist_camera", "right_wrist_camera"], #["left_wrist_camera", "right_wrist_camera", "front_camera",],
    help=(
        "Camera keys to send, in order. The default matches the ``video_keys`` "
        "ordering used by ``IsaacLabDataConfig`` "
        "(``[left_wrist_view, right_wrist_view, room_view]``), so ``--grid_mode`` "
        "produces the same 2x2 layout as ``VideoZImageAugmentTransform`` during "
        "RLinf training."
    ),
)
parser.add_argument(
    "--grid_mode",
    action="store_true",
    help=(
        "Tile up to 4 cameras into a single 2x2 grid before sending to Z-Image. "
        "The layout is ``[ cam0 | cam1 ]`` on top, ``[ cam2 | cam3 ]`` on bottom, "
        "following ``--cameras`` order; empty slots are filled with black."
    ),
)
parser.add_argument(
    "--inpaint_background",
    action="store_true",
    help=(
        "Send an inpaint mask to Z-Image so background pixels are regenerated "
        "while ``--inpaint_preserve_classes`` stay unchanged."
    ),
)
parser.add_argument(
    "--inpaint_preserve_classes",
    type=str,
    nargs="*",
    default=["robot", "trocar", "trocar_device", "tray", "cart"],
    help="Semantic class names to preserve unchanged when ``--inpaint_background`` is set.",
)
parser.add_argument(
    "--inpaint_preserve_dilation_px",
    type=int,
    default=0,
    help=(
        "Expand the preserved semantic foreground mask by this many pixels before "
        "inpainting. This reduces foreground-edge hallucinations such as extra "
        "arms attached to real grippers."
    ),
)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.z_image_view_size is not None:
    view_height, view_width = args_cli.z_image_view_size
    if view_height <= 0 or view_width <= 0:
        parser.error("--z_image_view_size values must be positive")
    args_cli.z_image_view_size = (view_height, view_width)

args_cli.enable_cameras = True
print(f"[test_z_image_augment] parsed args: {args_cli}", flush=True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[test_z_image_augment] AppLauncher up", flush=True)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import zmq  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

_TASK_DIR = Path(__file__).resolve().parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))
from _obs_helpers import (  # noqa: E402
    _ensure_tools_on_path,
    build_id_color_lut,
    compute_hold_action,
    resolve_fill_label_ids,
    seg_to_class_rgb,
    to_numpy,
)


def _save_uint8(arr: np.ndarray, path: Path) -> None:
    """Save ``(H, W, 3)`` uint8 array as RGB PNG."""
    from PIL import Image

    image = arr
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    Image.fromarray(image[..., :3]).save(path)


def _resize_frame(
    frame: np.ndarray,
    size_hw: tuple[int, int] | None,
    *,
    mode: str,
) -> np.ndarray:
    """Resize a ``(H, W, C)`` float frame to ``size_hw`` if requested."""
    if size_hw is None:
        return frame
    if frame.ndim != 3:
        raise ValueError(f"expected frame with shape (H, W, C), got {frame.shape}")

    height, width = size_hw
    if frame.shape[:2] == (height, width):
        return np.ascontiguousarray(frame)

    frame_t = torch.from_numpy(np.ascontiguousarray(np.transpose(frame, (2, 0, 1))))[None].float()
    if mode == "nearest":
        resized_t = torch.nn.functional.interpolate(frame_t, size=(height, width), mode="nearest")
    elif mode == "bilinear":
        resized_t = torch.nn.functional.interpolate(
            frame_t,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
    else:
        raise ValueError(f"unsupported resize mode: {mode}")

    resized = resized_t[0].permute(1, 2, 0).numpy()
    return np.ascontiguousarray(np.clip(resized, 0.0, 1.0).astype(np.float32))


def _build_grid_2x2(frames: list[np.ndarray]) -> np.ndarray:
    """Stack up to 4 ``(H, W, 3)`` frames into a ``(2H, 2W, 3)`` 2x2 grid."""
    if not frames:
        raise ValueError("_build_grid_2x2 requires at least one frame")
    if len(frames) > 4:
        raise ValueError(f"_build_grid_2x2 supports at most 4 frames, got {len(frames)}")
    height, width, channels = frames[0].shape
    for frame in frames[1:]:
        if frame.shape != (height, width, channels):
            raise ValueError(f"grid frames must share shape; got {frame.shape} vs {(height, width, channels)}")
    zero = np.zeros((height, width, channels), dtype=frames[0].dtype)
    slots = (frames + [zero] * 4)[:4]
    top = np.concatenate([slots[0], slots[1]], axis=1)
    bottom = np.concatenate([slots[2], slots[3]], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _split_grid_2x2(grid: np.ndarray, height: int, width: int, n_slots: int) -> list[np.ndarray]:
    """Split a ``(2H, 2W, 3)`` grid back into ``n_slots`` ``(H, W, 3)`` frames."""
    quadrants = [
        grid[:height, :width],
        grid[:height, width : 2 * width],
        grid[height : 2 * height, :width],
        grid[height : 2 * height, width : 2 * width],
    ]
    return [np.ascontiguousarray(q) for q in quadrants[:n_slots]]


def _prepare_rgb(rgb_group: dict[str, torch.Tensor], cam: str, env_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rgb_uint8, rgb_float01)`` for one camera at one env index."""
    rgb_np = to_numpy(rgb_group[cam][env_idx])
    if rgb_np.ndim != 3 or rgb_np.shape[-1] < 3:
        raise ValueError(f"expected RGB image with at least 3 channels for '{cam}', got {rgb_np.shape}")

    rgb_np = rgb_np[..., :3]
    if rgb_np.dtype == np.uint8:
        rgb_u8 = np.ascontiguousarray(rgb_np)
        rgb = rgb_u8.astype(np.float32) / 255.0
    else:
        rgb = np.ascontiguousarray(np.clip(rgb_np.astype(np.float32), 0.0, 1.0))
        rgb_u8 = (rgb * 255.0).astype(np.uint8)
    return rgb_u8, rgb


def _depth_to_z_image_rgb(depth: np.ndarray) -> np.ndarray:
    """Convert metric depth to VideoX-Fun/ZoeDepth-style RGB control.

    VideoX-Fun's ``ImageToDepth`` preprocessor normalizes depth with the 2nd
    and 85th percentiles, then inverts it so near surfaces are bright. Use the
    same shape convention for sim metric depth instead of Cosmos' reciprocal
    inverse-depth normalization.
    """
    arr = depth
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    valid = np.isfinite(arr) & (arr > 0)
    if not valid.any():
        return np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.float32)

    values = arr[valid]
    vmin = float(np.percentile(values, 2))
    vmax = float(np.percentile(values, 85))
    if vmax <= vmin + 1e-8:
        vmin = float(values.min())
        vmax = float(values.max())
    span = max(vmax - vmin, 1e-8)
    norm = (arr - vmin) / span
    depth_control = np.where(valid, 1.0 - norm, 0.0)
    depth_control = np.clip(depth_control, 0.0, 1.0).astype(np.float32)
    return np.repeat(depth_control[..., None], 3, axis=-1)


def _prepare_control(
    env,
    cam: str,
    env_idx: int,
    depth_group: dict[str, torch.Tensor],
    seg_group: dict[str, torch.Tensor],
    fill_ids_per_cam: dict[str, list[int]],
    control_kind: str,
) -> np.ndarray | None:
    """Build the optional Z-Image control frame for one camera."""
    if control_kind in {"edge", "none"}:
        return None
    if control_kind == "depth":
        if cam not in depth_group:
            raise RuntimeError(f"depth control requested but no depth obs for '{cam}'")
        depth_np = to_numpy(depth_group[cam][env_idx])
        return _depth_to_z_image_rgb(depth_np)
    if control_kind == "seg":
        if cam not in seg_group:
            raise RuntimeError(f"seg control requested but no seg obs for '{cam}'")
        seg_np = to_numpy(seg_group[cam][env_idx])
        lut = build_id_color_lut(env, cam)
        return seg_to_class_rgb(seg_np, id_color_lut=lut, fill_label_ids=fill_ids_per_cam.get(cam, []))
    raise ValueError(f"unsupported control_kind: {control_kind}")


def _dilate_binary_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Expand a binary mask by ``radius_px`` pixels."""
    if radius_px < 0:
        raise ValueError(f"inpaint preserve dilation must be non-negative, got {radius_px}")
    if radius_px == 0:
        return mask

    mask_t = torch.from_numpy(mask.astype(np.float32, copy=False))[None, None]
    kernel_size = 2 * radius_px + 1
    dilated = torch.nn.functional.max_pool2d(mask_t, kernel_size=kernel_size, stride=1, padding=radius_px)
    return (dilated[0, 0].numpy() > 0.5).astype(np.float32)


def _prepare_inpaint_mask(
    cam: str,
    env_idx: int,
    seg_group: dict[str, torch.Tensor],
    preserve_ids_per_cam: dict[str, list[int]],
    fill_ids_per_cam: dict[str, list[int]],
    preserve_dilation_px: int,
) -> np.ndarray:
    """Build an inpaint mask where white=regenerate background, black=preserve foreground."""
    if cam not in seg_group:
        raise RuntimeError(f"inpaint background requested but no seg obs for '{cam}'")
    preserve_ids = preserve_ids_per_cam.get(cam, [])
    if not preserve_ids:
        raise RuntimeError(f"inpaint background requested but no preserve class ids resolved for '{cam}'")

    seg_np = to_numpy(seg_group[cam][env_idx])
    if seg_np.ndim == 3 and seg_np.shape[-1] == 1:
        seg_np = seg_np[..., 0]
    seg_np = seg_np.astype(np.int32)

    fill_ids = fill_ids_per_cam.get(cam, [])
    active_fill_ids = [label_id for label_id in fill_ids if label_id in set(int(x) for x in np.unique(seg_np))]
    if active_fill_ids:
        _ensure_tools_on_path()
        from fill_trocar_mask import _fill_frame  # noqa: PLC0415

        seg_np = _fill_frame(seg_np, active_fill_ids)

    preserve = np.isin(seg_np, np.asarray(preserve_ids, dtype=np.int32)).astype(np.float32)
    preserve = _dilate_binary_mask(preserve, preserve_dilation_px)
    inpaint = 1.0 - preserve
    return np.repeat(inpaint[..., None], 3, axis=-1)


def _pack_tchw_block(tag: bytes, frame: np.ndarray) -> bytes:
    """Pack one ``(H, W, 3)`` float32 frame as a tagged CXG1 tensor block."""
    if len(tag) != 4:
        raise ValueError(f"CXG1 tag must be 4 bytes, got {tag!r}")
    height, width, channels = frame.shape
    if channels != 3:
        raise ValueError(f"expected 3-channel frame, got shape {frame.shape}")
    frame_tchw = np.ascontiguousarray(np.transpose(frame, (2, 0, 1))[None], dtype=np.float32)
    return tag + struct.pack("4I", 1, channels, height, width) + frame_tchw.tobytes()


def _z_image_request(
    sock: "zmq.Socket",
    rgb: np.ndarray,
    control: np.ndarray | None,
    inpaint_mask: np.ndarray | None,
    seed: int,
    disable_control: bool,
) -> np.ndarray:
    """Round-trip a single frame through the Z-Image service.

    Args:
        sock: REQ socket already connected to the service.
        rgb: ``(H, W, 3)`` float32 in ``[0, 1]``.
        control: Optional ``(H, W, 3)`` float32 control image in ``[0, 1]``.
        inpaint_mask: Optional ``(H, W, 3)`` float32 mask where white is regenerated.
        seed: Seed forwarded to the Z-Image sampler.
        disable_control: Whether to disable the service's internal edge-control fallback.

    Returns:
        ``(H, W, 3)`` float32 in ``[0, 1]`` containing the augmented RGB frame.
    """
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError(f"expected 3-channel RGB frame, got shape {rgb.shape}")

    rgb_tchw = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None], dtype=np.float32)
    payload = struct.pack("5I", 1, channels, height, width, seed) + rgb_tchw.tobytes()
    if control is not None and disable_control:
        raise ValueError("disable_control=True cannot be combined with an explicit control frame")

    blocks = []
    if disable_control:
        blocks.append(TRAILER_TAG_NO_CONTROL)
    if control is not None:
        if control.shape != rgb.shape:
            raise ValueError(f"control shape {control.shape} must match RGB shape {rgb.shape}")
        blocks.append(_pack_tchw_block(TRAILER_TAG_CONTROL, control))
    if inpaint_mask is not None:
        if inpaint_mask.shape != rgb.shape:
            raise ValueError(f"inpaint mask shape {inpaint_mask.shape} must match RGB shape {rgb.shape}")
        blocks.append(_pack_tchw_block(TRAILER_TAG_INPAINT_MASK, inpaint_mask))
    if blocks:
        payload += TRAILER_MAGIC + struct.pack("I", len(blocks)) + b"".join(blocks)

    sock.send(payload)
    reply = sock.recv()
    rT, rC, rH, rW = struct.unpack("4I", reply[:16])
    out = np.frombuffer(reply[16:], dtype=np.float32).copy().reshape(rT, rC, rH, rW)
    return np.transpose(out[0], (1, 2, 0))


def main() -> None:
    print("[test_z_image_augment] entered main()", flush=True)
    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=getattr(args_cli, "device", None) or "cuda:0",
        num_envs=args_cli.num_envs,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO] Task: {args_cli.task}  num_envs: {env.unwrapped.num_envs}", flush=True)
    print(f"[INFO] Output dir: {out_dir.resolve()}", flush=True)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, args_cli.z_image_timeout_s * 1000)
    sock.setsockopt(zmq.SNDTIMEO, args_cli.z_image_timeout_s * 1000)
    addr = f"tcp://{args_cli.z_image_host}:{args_cli.z_image_port}"
    sock.connect(addr)
    print(
        f"[INFO] Z-Image service endpoint: {addr}  control_kind={args_cli.control_kind}  "
        f"grid_mode={args_cli.grid_mode}  inpaint_background={args_cli.inpaint_background}  "
        f"view_size={args_cli.z_image_view_size or 'original'}",
        flush=True,
    )

    try:
        obs, _ = env.reset()
        hold_action = compute_hold_action(env)
        for i in range(max(args_cli.warmup_steps, 0)):
            obs, *_ = env.step(hold_action)
            print(f"[INFO] warmup step {i + 1}/{args_cli.warmup_steps}", flush=True)

        requested_cams = list(args_cli.cameras)
        fill_ids_per_cam = resolve_fill_label_ids(env, requested_cams, args_cli.fill_classes)
        preserve_ids_per_cam = (
            resolve_fill_label_ids(env, requested_cams, args_cli.inpaint_preserve_classes)
            if args_cli.inpaint_background
            else {}
        )
        print(f"[INFO] cameras={requested_cams}  fill_ids={fill_ids_per_cam}", flush=True)
        if args_cli.inpaint_background:
            print(
                f"[INFO] inpaint_preserve_classes={args_cli.inpaint_preserve_classes}  "
                f"preserve_ids={preserve_ids_per_cam}  "
                f"preserve_dilation_px={args_cli.inpaint_preserve_dilation_px}",
                flush=True,
            )

        for frame_idx in range(args_cli.num_frames):
            obs, *_ = env.step(hold_action)
            rgb_group: dict[str, torch.Tensor] = obs.get("camera_images", {})
            depth_group: dict[str, torch.Tensor] = obs.get("camera_depth", {})
            seg_group: dict[str, torch.Tensor] = obs.get("camera_semantic_segmentation", {})

            for env_idx in range(env.unwrapped.num_envs):
                env_dir = out_dir / f"env_{env_idx:02d}" / f"frame_{frame_idx:03d}"
                env_dir.mkdir(parents=True, exist_ok=True)

                per_cam_rgb_f: list[np.ndarray] = []
                per_cam_control: list[np.ndarray | None] = []
                per_cam_inpaint_mask: list[np.ndarray | None] = []
                used_cams: list[str] = []
                for cam in requested_cams:
                    if cam not in rgb_group:
                        print(f"[WARN] camera '{cam}' has no RGB obs; skipping", flush=True)
                        continue

                    rgb_u8, rgb = _prepare_rgb(rgb_group, cam, env_idx)
                    control = _prepare_control(
                        env, cam, env_idx, depth_group, seg_group, fill_ids_per_cam, args_cli.control_kind
                    )
                    inpaint_mask = (
                        _prepare_inpaint_mask(
                            cam,
                            env_idx,
                            seg_group,
                            preserve_ids_per_cam,
                            fill_ids_per_cam,
                            args_cli.inpaint_preserve_dilation_px,
                        )
                        if args_cli.inpaint_background
                        else None
                    )
                    send_rgb = _resize_frame(rgb, args_cli.z_image_view_size, mode="bilinear")
                    send_control = (
                        _resize_frame(control, args_cli.z_image_view_size, mode="bilinear")
                        if control is not None
                        else None
                    )
                    send_inpaint_mask = (
                        _resize_frame(inpaint_mask, args_cli.z_image_view_size, mode="nearest")
                        if inpaint_mask is not None
                        else None
                    )
                    per_cam_rgb_f.append(send_rgb)
                    per_cam_control.append(send_control)
                    per_cam_inpaint_mask.append(send_inpaint_mask)
                    used_cams.append(cam)
                    _save_uint8(rgb_u8, env_dir / f"{cam}_input_rgb.png")
                    if args_cli.z_image_view_size is not None:
                        _save_uint8((send_rgb * 255).astype(np.uint8), env_dir / f"{cam}_z_image_input_rgb.png")
                    if send_control is not None:
                        _save_uint8(
                            (send_control * 255).astype(np.uint8),
                            env_dir / f"{cam}_control_{args_cli.control_kind}.png",
                        )
                    if send_inpaint_mask is not None:
                        _save_uint8(
                            (send_inpaint_mask * 255).astype(np.uint8),
                            env_dir / f"{cam}_inpaint_mask.png",
                        )

                if not used_cams:
                    print(f"[WARN] env {env_idx}: no usable cameras this frame", flush=True)
                    continue

                if args_cli.grid_mode:
                    grid_rgb = _build_grid_2x2(per_cam_rgb_f)
                    grid_control = (
                        _build_grid_2x2(
                            [c if c is not None else np.zeros_like(per_cam_rgb_f[0]) for c in per_cam_control]
                        )
                        if any(c is not None for c in per_cam_control)
                        else None
                    )
                    grid_inpaint_mask = (
                        _build_grid_2x2(
                            [m if m is not None else np.zeros_like(per_cam_rgb_f[0]) for m in per_cam_inpaint_mask]
                        )
                        if args_cli.inpaint_background
                        else None
                    )
                    _save_uint8((grid_rgb * 255).astype(np.uint8), env_dir / "grid_input_rgb.png")
                    if grid_control is not None:
                        _save_uint8(
                            (grid_control * 255).astype(np.uint8),
                            env_dir / f"grid_control_{args_cli.control_kind}.png",
                        )
                    if grid_inpaint_mask is not None:
                        _save_uint8((grid_inpaint_mask * 255).astype(np.uint8), env_dir / "grid_inpaint_mask.png")

                    t0 = time.perf_counter()
                    seed = int(args_cli.z_image_seed + frame_idx * 1000 + env_idx)
                    augmented_grid = _z_image_request(
                        sock,
                        grid_rgb,
                        grid_control,
                        grid_inpaint_mask,
                        seed,
                        disable_control=args_cli.control_kind == "none",
                    )
                    dt = time.perf_counter() - t0
                    _save_uint8((augmented_grid * 255).astype(np.uint8), env_dir / "grid_z_image_out.png")

                    height, width = per_cam_rgb_f[0].shape[:2]
                    quadrants = _split_grid_2x2(augmented_grid, height, width, len(used_cams))
                    for cam, quadrant in zip(used_cams, quadrants):
                        _save_uint8((quadrant * 255).astype(np.uint8), env_dir / f"{cam}_z_image_out.png")
                    print(
                        f"[INFO] env={env_idx} frame={frame_idx} grid  z_image={dt:.2f}s  "
                        f"cameras={used_cams}  control_kind={args_cli.control_kind}  "
                        f"inpaint_background={args_cli.inpaint_background}",
                        flush=True,
                    )
                else:
                    for cam, rgb, control, inpaint_mask in zip(
                        used_cams, per_cam_rgb_f, per_cam_control, per_cam_inpaint_mask
                    ):
                        t0 = time.perf_counter()
                        seed = int(args_cli.z_image_seed + frame_idx * 1000 + env_idx)
                        augmented = _z_image_request(
                            sock,
                            rgb,
                            control,
                            inpaint_mask,
                            seed,
                            disable_control=args_cli.control_kind == "none",
                        )
                        dt = time.perf_counter() - t0
                        _save_uint8((augmented * 255).astype(np.uint8), env_dir / f"{cam}_z_image_out.png")
                        print(
                            f"[INFO] env={env_idx} frame={frame_idx} cam={cam}  z_image={dt:.2f}s  "
                            f"control_kind={args_cli.control_kind}  "
                            f"inpaint_background={args_cli.inpaint_background}",
                            flush=True,
                        )

        print(f"[DONE] Images written to {out_dir.resolve()}", flush=True)
    finally:
        try:
            sock.close(linger=0)
        finally:
            ctx.term()
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
