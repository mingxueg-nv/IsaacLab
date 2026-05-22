# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GR00T N1.7 modality config for IsaacLab assemble_trocar task (G1+Dex3).

What changed vs the N1.5 version of this file:

  * N1.7 removed ``gr00t.experiment.data_config`` (``BaseDataConfig`` /
    ``DATA_CONFIG_MAP``) and ``gr00t.model.transforms.GR00TTransform``. The
    entire transform pipeline (video crop / colour jitter / normalisation /
    padding to head width) is now baked into ``Gr00tN1d7Processor`` and is
    configured *on the ckpt side* via ``experiment_cfg/config.yaml`` at SFT
    time. Users only declare:

      1. which keys exist (``modality_keys``, ``delta_indices``),
      2. which action representation each key uses (``RELATIVE`` / ``ABSOLUTE``,
         ``EEF`` / ``NON_EEF``).

  * Normalisation mode is no longer set per-key here — it's read from
    ``statistics.json`` (``q01`` / ``q99`` percentiles) in the ckpt.

  * Action-side relative→absolute conversion is done by
    ``Gr00tN1d7Processor.decode_action`` automatically; this file does **not**
    need to express that math.

  * The ``IsaacLabDataConfig`` Python class is kept only as a tiny compatibility
    shim — the new RLinf N1.7 adapter (``rlinf.models.embodiment.gr00t_1_7``)
    loads ``modality_config`` directly from the ckpt's
    ``experiment_cfg/config.yaml`` and does **not** import this class. If
    anything in IsaacLab still references ``data_config_class:
    "gr00t_config:IsaacLabDataConfig"`` it will still resolve, but the returned
    config is the same dict we register below.

Source of truth for all numbers (ckpt
``/localhome/local-mingxueg/mingxue/models/GR00tN1_7_sim_assemble_trocar/
tune_visual/checkpoint-30000/experiment_cfg/config.yaml``):

    action.modality_keys     : [left_arm, right_arm, left_hand, right_hand]
    action.delta_indices     : 0..15  (16 chunks)
    action.action_configs    : [RELATIVE, RELATIVE, ABSOLUTE, ABSOLUTE]
                               (arms = delta-joint, hands = target-joint)
    state.modality_keys      : same 4 keys, 7 dims each (28 total)
    video.modality_keys      : [left_wrist_view, right_wrist_view, room_view]
    language.modality_keys   : [annotation.human.task_description]
    embodiment_tag           : NEW_EMBODIMENT (id 10 in N1.7 embodiment_id.json)
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# Bare modality keys (no "video." / "state." / "action." prefix — N1.7
# convention). The prefix is added downstream by Gr00tN1d7Processor.
VIDEO_KEYS = ["left_wrist_view", "right_wrist_view", "room_view"]
STATE_KEYS = ["left_arm", "right_arm", "left_hand", "right_hand"]
ACTION_KEYS = ["left_arm", "right_arm", "left_hand", "right_hand"]
LANGUAGE_KEYS = ["annotation.human.task_description"]

# Action chunk window: 16 steps. Must match ckpt action.delta_indices length.
ACTION_DELTA_INDICES = list(range(16))


ISAACLAB_G1_DEX3_MODALITY_CONFIG: dict[str, ModalityConfig] = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=VIDEO_KEYS,
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=STATE_KEYS,
    ),
    "action": ModalityConfig(
        delta_indices=ACTION_DELTA_INDICES,
        modality_keys=ACTION_KEYS,
        action_configs=[
            # left_arm: 7-DoF joint angles, trained as deltas → safer for
            # joint-space control; processor will do q_target = current + dq.
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_arm: same as left_arm.
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # left_hand (Dex3, 7-DoF): hands are easier to learn as absolute
            # target finger positions in our SFT recipe.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # right_hand: same as left_hand.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=LANGUAGE_KEYS,
    ),
}


# Register against NEW_EMBODIMENT so GR00T's processor / model heads route
# this embodiment through the correct projector slot (id=10 in the N1.7
# embodiment_id.json shipped with our SFT ckpt).
register_modality_config(
    ISAACLAB_G1_DEX3_MODALITY_CONFIG,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)


class IsaacLabDataConfig:
    """Compatibility shim for legacy N1.5 yaml ``data_config_class`` field.

    The RLinf N1.7 adapter (``rlinf.models.embodiment.gr00t_1_7.get_model``)
    pulls ``modality_config`` straight out of the ckpt's
    ``experiment_cfg/config.yaml``, so this class is effectively unused on the
    N1.7 path. Keeping the class around lets the N1.5 monkeypatch in
    ``isaaclab_contrib/rl/rlinf/extension.py:_patch_gr00t_get_model`` still
    resolve when someone runs the legacy yaml.
    """

    def modality_config(self) -> dict[str, ModalityConfig]:
        return ISAACLAB_G1_DEX3_MODALITY_CONFIG

    def transform(self):
        # N1.7 has no separate GR00TTransform — everything lives in
        # Gr00tN1d7Processor inside the model. RLinf's N1.7 adapter checks
        # for ``modality_transform is None`` and falls through to constructing
        # the processor from the ckpt files.
        return None
