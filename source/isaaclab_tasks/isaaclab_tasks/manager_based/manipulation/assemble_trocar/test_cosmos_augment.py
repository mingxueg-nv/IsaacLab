# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Send sim RGB + (optionally) sim depth/segmentation to the Cosmos-Transfer2.5
service and save the augmented output for visual inspection.

This mirrors ``save_camera_observations.py``: it runs the multimodal
assemble_trocar env, captures frames from each camera, and — instead of just
dumping PNGs — it ships each frame to a running ``cosmos_service.py`` process
over ZMQ and writes an ``input_rgb / control / cosmos_out`` triple per frame.

Prereqs (the service process must already be running):

    # one service per GPU, pick --control-type to match --control_kind below
    cd Isaac-GR00T/third_party/cosmos-transfer2.5
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python ../../scripts/cosmos_service.py \\
        --port 5557 --control-type depth

Usage::

    ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/test_cosmos_augment.py \\
        --num_envs 1 --num_frames 2 --control_kind depth --cosmos_port 5557 \\
        --guided_generation --output_dir outputs/cosmos_augment --headless

Notes:
    * ``--control_kind depth``: the depth control sent to Cosmos is the same
      inverse-depth-normalized grayscale as ``save_camera_observations.py``
      saves (near = bright), replicated to three channels.
    * ``--control_kind seg``: the control is the same view-consistent
      class-name-colored segmentation (using ``CLASS_COLORS``).
    * ``--control_kind none``: sends only RGB; Cosmos auto-estimates control
      internally. Useful as a baseline sanity check.
    * ``--guided_generation``: sends a binary foreground mask derived from sim
      semantic segmentation to Cosmos guided generation. White regions are
      anchored to the input frame while black regions remain free to change.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

TRAILER_MAGIC = b"CXG1"
TRAILER_TAG_CONTROL = b"CTRL"
TRAILER_TAG_GUIDED_MASK = b"GMAS"
TRAILER_TAG_GUIDED_CONFIG = b"GCFG"

parser = argparse.ArgumentParser(description="Send sim frames to Cosmos-Transfer2.5 and save augmented output.")
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
    default="outputs/cosmos_augment",
    help="Directory to write augmented PNGs to.",
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
    help="Semantic class names whose masks should get OmniGlass shaft-fill applied.",
)
parser.add_argument(
    "--control_kind",
    type=str,
    default="depth",
    choices=["depth", "seg", "none"],
    help=(
        "Which sim control signal to ship to the Cosmos service. "
        "``none`` sends RGB only and Cosmos auto-estimates the control."
    ),
)
parser.add_argument("--cosmos_host", type=str, default="localhost", help="Cosmos service ZMQ host.")
parser.add_argument("--cosmos_port", type=int, default=5557, help="Cosmos service ZMQ port.")
parser.add_argument(
    "--cosmos_seed",
    type=int,
    default=0,
    help="Seed for the Cosmos generation (same seed → deterministic output for the same inputs).",
)
parser.add_argument(
    "--cosmos_timeout_s",
    type=int,
    default=300,
    help="Per-request timeout for the Cosmos service (seconds).",
)
parser.add_argument(
    "--cameras",
    type=str,
    nargs="*",
    default=["left_wrist_camera", "right_wrist_camera", "front_camera"],
    help=(
        "Camera keys to send, in order. The default matches the ``video_keys`` "
        "ordering used by ``IsaacLabDataConfig`` "
        "(``[left_wrist_view, right_wrist_view, room_view]``) so ``--grid_mode`` "
        "produces the same 2x2 layout as ``VideoCosmosAugmentTransform`` during "
        "RLinf training."
    ),
)
parser.add_argument(
    "--grid_mode",
    action="store_true",
    help=(
        "Tile up to 4 cameras into a single 2x2 grid before sending to Cosmos, "
        "mirroring VideoCosmosAugmentTransform(grid_mode=True). The grid layout "
        "is ``[ cam0 | cam1 ]`` on top, ``[ cam2 | cam3 ]`` on bottom, following "
        "``--cameras`` order; empty slots are filled with black. The service "
        "produces one augmented grid per call, which is split back into per-camera "
        "outputs. Yields spatially coherent augmentation across views and reduces "
        "Cosmos calls by ~3x when all three cameras are used."
    ),
)
parser.add_argument(
    "--guided_generation",
    action="store_true",
    help=(
        "Send a binary guided-generation mask to the Cosmos service. The mask is "
        "derived from semantic segmentation and marks foreground classes as white "
        "so Cosmos preserves their input-frame structure during denoising."
    ),
)
parser.add_argument(
    "--guided_classes",
    type=str,
    nargs="*",
    default=["robot", "trocar", "trocar_device", "tray", "cart", "instrument_trolley"],
    help="Semantic class names to mark as foreground for ``--guided_generation``.",
)
parser.add_argument(
    "--guided_generation_step_threshold",
    type=int,
    default=25,
    help="Number of denoising steps that use the guided-generation mask in Cosmos.",
)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
print(f"[test_cosmos_augment] parsed args: {args_cli}", flush=True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[test_cosmos_augment] AppLauncher up", flush=True)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import zmq  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

# Reuse sim-side helpers so the control signals sent to Cosmos match the ones
# ``save_camera_observations.py`` saves exactly.
_TASK_DIR = Path(__file__).resolve().parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))
from _obs_helpers import (  # noqa: E402
    build_id_color_lut,
    compute_hold_action,
    depth_to_inverse_rgb,
    resolve_fill_label_ids,
    seg_to_class_rgb,
    to_numpy,
)


def _save_uint8(arr: np.ndarray, path: Path) -> None:
    """Save ``(H, W, 3)`` uint8 array as RGB PNG."""
    from PIL import Image

    a = arr
    if a.dtype != np.uint8:
        a = np.clip(a, 0.0, 1.0)
        a = (a * 255.0).astype(np.uint8)
    Image.fromarray(a[..., :3], mode="RGB").save(path)


def _build_grid_2x2(frames: list[np.ndarray]) -> np.ndarray:
    """Stack up to 4 ``(H, W, 3)`` frames into a ``(2H, 2W, 3)`` 2x2 grid.

    Layout mirrors ``VideoCosmosAugmentTransform._build_grid``::

        [ slots[0] | slots[1] ]
        [ slots[2] | slots[3] ]

    Missing slots are filled with zeros, so three cameras produce a grid with
    a solid-black bottom-right quadrant (same as the RLinf training path).
    """
    if not frames:
        raise ValueError("_build_grid_2x2 requires at least one frame")
    H, W, C = frames[0].shape
    for f in frames[1:]:
        if f.shape != (H, W, C):
            raise ValueError(f"grid frames must share shape; got {f.shape} vs {(H, W, C)}")
    zero = np.zeros((H, W, C), dtype=frames[0].dtype)
    slots = (frames + [zero] * 4)[:4]
    top = np.concatenate([slots[0], slots[1]], axis=1)      # (H, 2W, C)
    bottom = np.concatenate([slots[2], slots[3]], axis=1)   # (H, 2W, C)
    return np.concatenate([top, bottom], axis=0)             # (2H, 2W, C)


def _split_grid_2x2(grid: np.ndarray, H: int, W: int, n_slots: int) -> list[np.ndarray]:
    """Split a ``(2H, 2W, 3)`` grid back into ``n_slots`` ``(H, W, 3)`` frames."""
    quadrants = [
        grid[:H, :W],
        grid[:H, W:2 * W],
        grid[H:2 * H, :W],
        grid[H:2 * H, W:2 * W],
    ]
    return [np.ascontiguousarray(q) for q in quadrants[:n_slots]]


def _prepare_control(
    env,
    cam: str,
    env_idx: int,
    depth_group: dict,
    seg_group: dict,
    fill_ids_per_cam: dict,
    control_kind: str,
) -> np.ndarray | None:
    """Build the Cosmos control frame for one camera. Returns ``(H, W, 3)`` float32 in [0, 1]."""
    if control_kind == "depth":
        if cam not in depth_group:
            raise RuntimeError(f"depth control requested but no depth obs for '{cam}'")
        depth_np = to_numpy(depth_group[cam][env_idx])
        return depth_to_inverse_rgb(depth_np)
    if control_kind == "seg":
        if cam not in seg_group:
            raise RuntimeError(f"seg control requested but no seg obs for '{cam}'")
        seg_np = to_numpy(seg_group[cam][env_idx])
        lut = build_id_color_lut(env, cam)
        return seg_to_class_rgb(seg_np, id_color_lut=lut, fill_label_ids=fill_ids_per_cam.get(cam, []))
    return None  # control_kind == "none"


def _prepare_guided_mask(
    cam: str,
    env_idx: int,
    seg_group: dict,
    guided_ids_per_cam: dict[str, list[int]],
) -> np.ndarray:
    """Build a binary RGB guided-generation mask for one camera.

    Returns:
        ``(H, W, 3)`` float32 in ``[0, 1]`` where foreground classes are white.
    """
    if cam not in seg_group:
        raise RuntimeError(f"guided generation requested but no seg obs for '{cam}'")
    label_ids = guided_ids_per_cam.get(cam, [])
    if not label_ids:
        raise RuntimeError(f"guided generation requested but no guided class ids resolved for '{cam}'")
    seg_np = to_numpy(seg_group[cam][env_idx])
    if seg_np.ndim == 3 and seg_np.shape[-1] == 1:
        seg_np = seg_np[..., 0]
    mask = np.isin(seg_np.astype(np.int32), np.asarray(label_ids, dtype=np.int32)).astype(np.float32)
    return np.repeat(mask[..., None], 3, axis=-1)


def _prepare_rgb(rgb_group: dict, cam: str, env_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rgb_uint8, rgb_float01)`` for one camera at one env index."""
    rgb_u8 = to_numpy(rgb_group[cam][env_idx])
    if rgb_u8.dtype != np.uint8 and rgb_u8.shape[-1] == 3:
        rgb_u8 = np.clip(rgb_u8, 0.0, 1.0)
        rgb_u8 = (rgb_u8 * 255.0).astype(np.uint8)
    rgb = rgb_u8[..., :3].astype(np.float32) / 255.0
    return rgb_u8, rgb


def _pack_tchw_block(tag: bytes, frame: np.ndarray) -> bytes:
    """Pack one ``(H, W, 3)`` float32 frame as a tagged CXG1 tensor block."""
    if len(tag) != 4:
        raise ValueError(f"CXG1 tag must be 4 bytes, got {tag!r}")
    H, W, C = frame.shape
    if C != 3:
        raise ValueError(f"expected 3-channel frame, got shape {frame.shape}")
    frame_tchw = np.ascontiguousarray(np.transpose(frame, (2, 0, 1))[None], dtype=np.float32)
    return tag + struct.pack("4I", 1, C, H, W) + frame_tchw.tobytes()


def _cosmos_request(
    sock: "zmq.Socket",
    rgb: np.ndarray,
    control: np.ndarray | None,
    guided_mask: np.ndarray | None,
    guided_generation_step_threshold: int,
    seed: int,
) -> np.ndarray:
    """Round-trip a single frame through the Cosmos service.

    Args:
        sock: REQ socket already connected to the service.
        rgb: ``(H, W, 3)`` float32 in ``[0, 1]``.
        control: Optional ``(H, W, 3)`` float32 in ``[0, 1]``. ``None`` → service
            auto-estimates the control signal.
        guided_mask: Optional binary RGB guided-generation mask in ``[0, 1]``.
        guided_generation_step_threshold: Number of denoising steps that use the mask.
        seed: Seed forwarded to Cosmos' sampler.

    Returns:
        ``(H, W, 3)`` float32 in ``[0, 1]`` (the augmented RGB frame).
    """
    H, W, _ = rgb.shape
    T, C = 1, 3
    # wire layout: (T, C, H, W, seed) + rgb float32 [T, C, H, W]
    rgb_tchw = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None], dtype=np.float32)
    payload = struct.pack("5I", T, C, H, W, seed) + rgb_tchw.tobytes()
    if guided_mask is not None:
        blocks = []
        if control is not None:
            if control.shape != rgb.shape:
                raise ValueError(f"control shape {control.shape} must match RGB shape {rgb.shape}")
            blocks.append(_pack_tchw_block(TRAILER_TAG_CONTROL, control))
        if guided_mask.shape != rgb.shape:
            raise ValueError(f"guided mask shape {guided_mask.shape} must match RGB shape {rgb.shape}")
        blocks.append(_pack_tchw_block(TRAILER_TAG_GUIDED_MASK, guided_mask))
        blocks.append(TRAILER_TAG_GUIDED_CONFIG + struct.pack("I", guided_generation_step_threshold))
        payload += TRAILER_MAGIC + struct.pack("I", len(blocks)) + b"".join(blocks)
    elif control is not None:
        if control.shape != rgb.shape:
            raise ValueError(f"control shape {control.shape} must match RGB shape {rgb.shape}")
        ctl_tchw = np.ascontiguousarray(np.transpose(control, (2, 0, 1))[None], dtype=np.float32)
        payload += struct.pack("4I", T, C, H, W) + ctl_tchw.tobytes()

    sock.send(payload)
    reply = sock.recv()
    rT, rC, rH, rW = struct.unpack("4I", reply[:16])
    out = np.frombuffer(reply[16:], dtype=np.float32).copy().reshape(rT, rC, rH, rW)
    # Return first frame as (H, W, 3).
    return np.transpose(out[0], (1, 2, 0))


def main() -> None:
    print("[test_cosmos_augment] entered main()", flush=True)
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
    sock.setsockopt(zmq.RCVTIMEO, args_cli.cosmos_timeout_s * 1000)
    sock.setsockopt(zmq.SNDTIMEO, args_cli.cosmos_timeout_s * 1000)
    addr = f"tcp://{args_cli.cosmos_host}:{args_cli.cosmos_port}"
    sock.connect(addr)
    print(
        f"[INFO] Cosmos service endpoint: {addr}  control_kind={args_cli.control_kind}  "
        f"guided_generation={args_cli.guided_generation}",
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
        guided_ids_per_cam = (
            resolve_fill_label_ids(env, requested_cams, args_cli.guided_classes)
            if args_cli.guided_generation
            else {}
        )
        print(f"[INFO] cameras={requested_cams}  fill_ids={fill_ids_per_cam}", flush=True)
        if args_cli.guided_generation:
            print(
                f"[INFO] guided_classes={args_cli.guided_classes}  "
                f"guided_ids={guided_ids_per_cam}  "
                f"guided_generation_step_threshold={args_cli.guided_generation_step_threshold}",
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

                # Gather inputs per camera in --cameras order.
                per_cam_rgb_u8: list[np.ndarray] = []
                per_cam_rgb_f: list[np.ndarray] = []
                per_cam_control: list[np.ndarray | None] = []
                per_cam_guided_mask: list[np.ndarray | None] = []
                used_cams: list[str] = []
                for cam in requested_cams:
                    if cam not in rgb_group:
                        print(f"[WARN] camera '{cam}' has no RGB obs; skipping", flush=True)
                        continue
                    rgb_u8, rgb = _prepare_rgb(rgb_group, cam, env_idx)
                    control = _prepare_control(
                        env, cam, env_idx, depth_group, seg_group, fill_ids_per_cam, args_cli.control_kind
                    )
                    guided_mask = (
                        _prepare_guided_mask(cam, env_idx, seg_group, guided_ids_per_cam)
                        if args_cli.guided_generation
                        else None
                    )
                    per_cam_rgb_u8.append(rgb_u8)
                    per_cam_rgb_f.append(rgb)
                    per_cam_control.append(control)
                    per_cam_guided_mask.append(guided_mask)
                    used_cams.append(cam)

                    # Always save per-camera inputs so the user can eyeball them.
                    _save_uint8(rgb_u8, env_dir / f"{cam}_input_rgb.png")
                    if control is not None:
                        ctl_name = f"{cam}_control_{args_cli.control_kind}.png"
                        _save_uint8((control * 255).astype(np.uint8), env_dir / ctl_name)
                    if guided_mask is not None:
                        _save_uint8((guided_mask * 255).astype(np.uint8), env_dir / f"{cam}_guided_mask.png")

                if not used_cams:
                    print(f"[WARN] env {env_idx}: no usable cameras this frame", flush=True)
                    continue

                if args_cli.grid_mode:
                    # Build and send a single 2x2 grid, then split the response back.
                    grid_rgb = _build_grid_2x2(per_cam_rgb_f)
                    grid_ctl = (
                        _build_grid_2x2([c if c is not None else np.zeros_like(per_cam_rgb_f[0])
                                         for c in per_cam_control])
                        if args_cli.control_kind != "none"
                        else None
                    )
                    grid_guided_mask = (
                        _build_grid_2x2(
                            [m if m is not None else np.zeros_like(per_cam_rgb_f[0]) for m in per_cam_guided_mask]
                        )
                        if args_cli.guided_generation
                        else None
                    )
                    _save_uint8((grid_rgb * 255).astype(np.uint8), env_dir / "grid_input_rgb.png")
                    if grid_ctl is not None:
                        _save_uint8((grid_ctl * 255).astype(np.uint8),
                                    env_dir / f"grid_control_{args_cli.control_kind}.png")
                    if grid_guided_mask is not None:
                        _save_uint8((grid_guided_mask * 255).astype(np.uint8), env_dir / "grid_guided_mask.png")

                    t0 = time.perf_counter()
                    seed = int(args_cli.cosmos_seed + frame_idx * 1000 + env_idx)
                    augmented_grid = _cosmos_request(
                        sock,
                        grid_rgb,
                        grid_ctl,
                        grid_guided_mask,
                        args_cli.guided_generation_step_threshold,
                        seed,
                    )
                    dt = time.perf_counter() - t0
                    _save_uint8((augmented_grid * 255).astype(np.uint8), env_dir / "grid_cosmos_out.png")

                    H, W = per_cam_rgb_f[0].shape[:2]
                    quads = _split_grid_2x2(augmented_grid, H, W, len(used_cams))
                    for cam, quad in zip(used_cams, quads):
                        _save_uint8((quad * 255).astype(np.uint8), env_dir / f"{cam}_cosmos_out.png")
                    print(
                        f"[INFO] env={env_idx} frame={frame_idx} grid  cosmos={dt:.2f}s  "
                        f"cameras={used_cams}  control_kind={args_cli.control_kind}  "
                        f"guided_generation={args_cli.guided_generation}",
                        flush=True,
                    )
                else:
                    # Per-camera: one ZMQ round-trip per camera.
                    for cam, rgb, control, guided_mask in zip(
                        used_cams, per_cam_rgb_f, per_cam_control, per_cam_guided_mask
                    ):
                        t0 = time.perf_counter()
                        seed = int(args_cli.cosmos_seed + frame_idx * 1000 + env_idx)
                        augmented = _cosmos_request(
                            sock,
                            rgb,
                            control,
                            guided_mask,
                            args_cli.guided_generation_step_threshold,
                            seed,
                        )
                        dt = time.perf_counter() - t0
                        _save_uint8((augmented * 255).astype(np.uint8), env_dir / f"{cam}_cosmos_out.png")
                        print(
                            f"[INFO] env={env_idx} frame={frame_idx} cam={cam}  "
                            f"cosmos={dt:.2f}s  control_kind={args_cli.control_kind}  "
                            f"guided_generation={args_cli.guided_generation}",
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
        sys.exit(0)
