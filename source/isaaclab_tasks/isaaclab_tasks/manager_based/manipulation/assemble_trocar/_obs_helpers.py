# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared helpers used by ``save_camera_observations.py`` and
``test_cosmos_augment.py``.

These live in a module separate from the scripts so both tools can import them
without triggering the scripts' top-level ``AppLauncher`` boot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# View-consistent class-name → RGB palette. Each camera has its own integer
# label allocation (``idToLabels`` is per-render-product), so we colorize
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


def _tools_dir() -> Path:
    """Return the repo's ``scripts/tools`` directory (for fill_trocar_mask import)."""
    return Path(__file__).resolve().parents[6] / "scripts" / "tools"


def _ensure_tools_on_path() -> None:
    tools = str(_tools_dir())
    if tools not in sys.path:
        sys.path.insert(0, tools)


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().contiguous().numpy()


def iter_camera_obs(obs_group: dict[str, torch.Tensor]):
    """Yield (camera_name, tensor) pairs for tensor entries in an obs group."""
    for key, value in obs_group.items():
        if isinstance(value, torch.Tensor):
            yield key, value


def compute_hold_action(env) -> torch.Tensor:
    """Build an action tensor that commands the robot to hold its current pose.

    The ``joint_pos`` action term applies ``targets = scale * action + offset``,
    so to hold we set ``action = (current_joint_pos - offset) / scale``. Handles
    Warp-backed joint_pos tensors transparently.
    """
    import warp as wp

    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    robot = env.unwrapped.scene["robot"]
    joint_pos = robot.data.joint_pos
    if not isinstance(joint_pos, torch.Tensor):
        joint_pos = wp.to_torch(joint_pos)

    joint_ids = action_term._joint_ids
    if isinstance(joint_ids, slice):
        selected = joint_pos[:, joint_ids]
    else:
        ids_tensor = torch.as_tensor(list(joint_ids), dtype=torch.long, device=joint_pos.device)
        selected = joint_pos.index_select(dim=1, index=ids_tensor)

    offset = action_term._offset
    scale = action_term._scale
    if isinstance(offset, torch.Tensor):
        offset = offset.to(selected.device)
    if isinstance(scale, torch.Tensor):
        scale = scale.to(selected.device)
    action = (selected - offset) / scale
    return action.contiguous()


def build_id_color_lut(env, camera_name: str) -> np.ndarray:
    """Return an ``(N, 3)`` uint8 LUT mapping integer label IDs → stable RGB.

    Colors are looked up by class *name* in :data:`CLASS_COLORS` so the same
    class renders with the same color in every camera view, even though each
    camera allocates its own numeric IDs.
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


def resolve_fill_label_ids(env, camera_names: list[str], class_names: list[str]) -> dict[str, list[int]]:
    """Look up integer label IDs for ``class_names`` per camera (via ``idToLabels``)."""
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


def depth_to_inverse_rgb(depth: np.ndarray) -> np.ndarray:
    """Convert metric depth (H, W) → 3-channel float32 [0, 1] inverse-depth RGB.

    Near = bright, far = dark. Matches the PNG writer in
    ``save_camera_observations.py`` and Cosmos-Transfer2.5's depth convention.
    """
    arr = depth
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    valid = np.isfinite(arr) & (arr > 0)
    if not valid.any():
        return np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.float32)
    inv = np.zeros_like(arr, dtype=np.float32)
    inv[valid] = 1.0 / arr[valid]
    vmin = float(inv[valid].min())
    vmax = float(inv[valid].max())
    span = max(vmax - vmin, 1e-8)
    norm = np.where(valid, (inv - vmin) / span, 0.0).astype(np.float32)
    norm = np.clip(norm, 0.0, 1.0)
    return np.repeat(norm[..., None], 3, axis=-1)


def seg_to_class_rgb(
    seg: np.ndarray,
    id_color_lut: np.ndarray,
    fill_label_ids: list[int],
) -> np.ndarray:
    """Convert int32 class-id map → 3-channel float32 [0, 1] RGB using the
    view-consistent ``CLASS_COLORS`` palette, with trocar-shaft fill applied."""
    _ensure_tools_on_path()
    from fill_trocar_mask import _fill_frame  # noqa: PLC0415

    arr = seg
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    arr = arr.astype(np.int32)

    present = set(int(x) for x in np.unique(arr))
    active = [lid for lid in fill_label_ids if lid in present]
    if active:
        arr = _fill_frame(arr, active)

    H, W = arr.shape
    rgb = np.full((H, W, 3), UNKNOWN_CLASS_COLOR, dtype=np.uint8)
    if id_color_lut.shape[0] > 0:
        in_range = (arr >= 0) & (arr < id_color_lut.shape[0])
        rgb[in_range] = id_color_lut[arr[in_range]]
    return rgb.astype(np.float32) / 255.0
