# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path

from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.utils.prims import clone
from isaaclab.sim.utils.semantics import add_labels
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.assemble_trocar import mdp

from isaaclab_tasks.manager_based.manipulation.assemble_trocar.config import (  # isort: skip
    CameraPresets,
    G1RobotPresets,
)

joint_names = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]
offset_dict = {
    "left_elbow_joint": -0.3,
    "right_elbow_joint": -0.3,
}

HEALTHCARE_S3 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/Healthcare/0.5.0/132c82d"
USD_ROOT = f"{HEALTHCARE_S3}/Props/LightWheel"
OMNIVERSE_PROPS_ROOT = "omniverse://isaac-dev.ov.nvidia.com/Library/IsaacHealthcare/0.5.0/Props"
REPO_ROOT = Path(__file__).resolve().parents[6]
LOCAL_USD_ROOT = os.environ.get(
    "ASSEMBLE_TROCAR_USD_ROOT",
    str(REPO_ROOT / "assets" / "assemble_trocar"),
)
SCENE_VARIANT = os.environ.get("ASSEMBLE_TROCAR_SCENE_VARIANT", "default").strip().lower().replace("-", "_")
SCENE_VARIANT_ALIASES = {
    "": "default",
    "default": "default",
    "scene03": "default",
    "lightwheel": "default",
    "factory": "factory",
    "orca": "orca",
    "orca_scene": "orca",
    "surgical": "surgical_room",
    "surgical_room": "surgical_room",
}
SCENE_VARIANT = SCENE_VARIANT_ALIASES.get(SCENE_VARIANT, SCENE_VARIANT)
SUPPORTED_SCENE_VARIANTS = tuple(sorted(set(SCENE_VARIANT_ALIASES.values())))
if SCENE_VARIANT not in SUPPORTED_SCENE_VARIANTS:
    raise ValueError(
        f"Unsupported ASSEMBLE_TROCAR_SCENE_VARIANT={SCENE_VARIANT!r}. "
        f"Expected one of {SUPPORTED_SCENE_VARIANTS}."
    )

SCENE_VARIANT_ENV_SPACING = {
    "default": 6.0,
    # Match the i4h benchmark background commits. These room-scale USDs need a
    # large spacing so their background/task-table geometry does not leak across
    # tiled vectorized renders.
    "factory": 50.0,
    "orca": 50.0,
    "surgical_room": 50.0,
}
SCENE_RANDOMIZE_LIGHTING_REQUESTED = (
    os.environ.get("ASSEMBLE_TROCAR_RANDOMIZE_LIGHTING", "false").strip().lower() in {"1", "true", "yes", "on"}
)
default_scene_env_spacing = SCENE_VARIANT_ENV_SPACING[SCENE_VARIANT]
SCENE_ENV_SPACING = float(
    os.environ.get("ASSEMBLE_TROCAR_ENV_SPACING", default_scene_env_spacing)
)
SCENE_REPLICATE_PHYSICS = os.environ.get(
    "ASSEMBLE_TROCAR_REPLICATE_PHYSICS",
    "true" if SCENE_VARIANT == "default" else "false",
).strip().lower() in {"1", "true", "yes", "on"}
# SCENE_REPLICATE_PHYSICS = False


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float_pair(name: str, default: tuple[float, float]) -> tuple[float, float]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{name} must be two comma-separated floats, got {value!r}.")
    return (float(parts[0]), float(parts[1]))


DEFAULT_SCENE_RANDOMIZE_LIGHTING = SCENE_RANDOMIZE_LIGHTING_REQUESTED and SCENE_VARIANT == "default"
DEFAULT_RANDOM_IMAGE_BRIGHTNESS_RANGE = _env_float_pair(
    "ASSEMBLE_TROCAR_RANDOM_IMAGE_BRIGHTNESS_RANGE",
    (30.0, 120.0),
)
DEFAULT_RANDOM_IMAGE_BRIGHTNESS_PRINT_LOG = _env_bool("ASSEMBLE_TROCAR_RANDOM_IMAGE_BRIGHTNESS_PRINT_LOG", False)
CAMERA_IMAGE_OBS_FUNC = mdp.image_with_random_brightness if DEFAULT_SCENE_RANDOMIZE_LIGHTING else base_mdp.image


def _wxyz_to_xyzw(rot: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Convert benchmark scene rotations from WXYZ into IsaacLab 6 XYZW."""
    return (rot[1], rot[2], rot[3], rot[0])


def _make_trocar_1_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/trocar_1",
        spawn=UsdFileCfg(
            # usd_path=f"{LOCAL_USD_ROOT}/Assets/Trocar002/Trocar004_test.usd",
            usd_path=f"{LOCAL_USD_ROOT}/Assets/Trocar002/Trocar004_test_nodeform.usd",
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.001,
                rest_offset=-0.001,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-1.60202, 1.91362, 0.87183],
            rot=[-0.0, 0.70711, 0.70711, 0.0],
        ),
    )


def _make_trocar_2_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/trocar_2",
        spawn=UsdFileCfg(
            usd_path=(
                f"{LOCAL_USD_ROOT}/Assets/"
                "DisposableLaparoscopicPunctureDevice001/"
                "DisposableLaparoscopicPunctureDevice006_test.usd"
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            rot=[-0.71475, -0.000243, 0.05853, 0.69692], pos=[-1.50635, 1.90997, 0.8631]
        ),
    )


def _make_tray_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/surgical_tray",
        spawn=UsdFileCfg(
            usd_path=f"{USD_ROOT}/Assets/SurgicalTray001/SurgicalTray001.usd",
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[-1.54919, 2.03365, 0.84554], rot=[0.0, 0.0, -0.70711, 0.70711]
        ),
        actuators={},  # Empty dict for passive articulation (no motors)
    )


def _make_cart001_cfg() -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="/World/envs/env_.*/cart001",
        spawn=UsdFileCfg(
            usd_path=f"{OMNIVERSE_PROPS_ROOT}/LightWheel/Assets/Cart001/Cart001.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-1.48242, 2.03195, 0.00279),
            rot=_wxyz_to_xyzw((1.0, 0.0, 0.0, 0.0)),
        ),
    )


def _make_instrument_trolley002_cfg() -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="/World/envs/env_.*/instrument_trolley002",
        spawn=UsdFileCfg(
            usd_path=f"{OMNIVERSE_PROPS_ROOT}/LightWheel/Assets/InstrumentTrolley001/InstrumentTrolley002.usd",
            scale=(1.05, 1.05, 1.05),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-1.52131, 1.4862, 0.0),
            rot=_wxyz_to_xyzw((0.0, 0.0, 0.0, 1.0)),
        ),
    )


def _make_default_light_cfg() -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=1000.0,
        ),
    )


def _make_surgical_room_light_cfg() -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=1000.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-3.8, 5.3, 2.0),
            rot=_wxyz_to_xyzw((1.0, 0.0, 0.0, 0.0)),
        ),
    )


def _make_surgical_room_tray_fill_light_cfg() -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path="/World/envs/env_.*/surgical_room_tray_fill_light",
        spawn=sim_utils.DiskLightCfg(
            color=(0.78, 0.90, 1.0),
            intensity=250.0,
            radius=1.35,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-1.55, 1.86, 1.55),
            rot=_wxyz_to_xyzw((1.0, 0.0, 0.0, 0.0)),
        ),
    )

@configclass
class AssembleTrocarSceneBaseCfg(InteractiveSceneCfg):
    """Shared scene configuration for the assemble_trocar task."""

    # humanoid robot configuration
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_base_fix(
        init_pos=(-1.84919, 1.94, 0.81168), init_rot=(0.0, 0.0, 0.0, 1.0)
    )
    # add camera configuration
    front_camera = CameraPresets.g1_front_camera()
    left_wrist_camera = CameraPresets.left_dex3_wrist_camera()
    right_wrist_camera = CameraPresets.right_dex3_wrist_camera()

    trocar_1 = _make_trocar_1_cfg()
    trocar_2 = _make_trocar_2_cfg()
    tray = _make_tray_cfg()


@configclass
class AssembleTrocarSceneCfg(AssembleTrocarSceneBaseCfg):
    """Default LightWheel scene03 background used by existing experiments."""

    scene = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Scene",
        spawn=UsdFileCfg(
            usd_path=f"{USD_ROOT}/scene03.usd",
        ),
    )
    light = _make_default_light_cfg()


@configclass
class AssembleTrocarFactorySceneCfg(AssembleTrocarSceneBaseCfg):
    """Factory background from i4h benchmark commit e67b3b8."""

    scene = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Scene",
        spawn=UsdFileCfg(
            usd_path=f"{OMNIVERSE_PROPS_ROOT}/OrcaScenes/Scene1MX2/rlinf_scenes/factory.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(1.0, 3.0, 0.0),
            rot=_wxyz_to_xyzw((0.0, 0.0, 0.0, 1.0)),
        ),
    )
    cart001 = _make_cart001_cfg()
    instrument_trolley002 = _make_instrument_trolley002_cfg()


@configclass
class AssembleTrocarOrcaSceneCfg(AssembleTrocarSceneBaseCfg):
    """Orca room background from i4h benchmark commit 98b6e4a."""

    scene = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Scene",
        spawn=UsdFileCfg(
            usd_path=f"{OMNIVERSE_PROPS_ROOT}/OrcaScenes/Scene1MX2/main_new_light.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(4.0, -5.5, 0.0),
            rot=_wxyz_to_xyzw((1.0, 0.0, 0.0, 0.0)),
        ),
    )
    cart001 = _make_cart001_cfg()
    instrument_trolley002 = _make_instrument_trolley002_cfg()


@configclass
class AssembleTrocarSurgicalRoomSceneCfg(AssembleTrocarSceneBaseCfg):
    """Surgical-room background from i4h benchmark commit 03480cf."""

    scene = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Scene",
        spawn=UsdFileCfg(
            usd_path=f"{OMNIVERSE_PROPS_ROOT}/OrcaScenes/Scene1MX2/push-cart-OR-scenes/main.usd",
            scale=(0.008, 0.008, 0.008),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-3.8, 5.3, 0.0),
            rot=_wxyz_to_xyzw((0.70710678, 0.0, 0.0, -0.70710678)),
        ),
    )
    cart001 = _make_cart001_cfg()
    instrument_trolley002 = _make_instrument_trolley002_cfg()
    light = _make_surgical_room_light_cfg()
    tray_fill_light = _make_surgical_room_tray_fill_light_cfg()


SCENE_VARIANT_TO_CFG = {
    "default": AssembleTrocarSceneCfg,
    "factory": AssembleTrocarFactorySceneCfg,
    "orca": AssembleTrocarOrcaSceneCfg,
    "surgical_room": AssembleTrocarSurgicalRoomSceneCfg,
}
ACTIVE_SCENE_CFG_CLS = SCENE_VARIANT_TO_CFG[SCENE_VARIANT]


##
# MDP settings
##
@configclass
class ActionsCfg:
    """defines the action configuration related to robot control, using direct joint angle control"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=joint_names,
        scale=1.0,
        use_default_offset=False,
        offset=offset_dict,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """defines all available observation information"""

    @configclass
    class PolicyCfg(ObsGroup):
        """policy group observation configuration class
        defines all state observation values for policy decision
        inherit from ObsGroup base class
        """

        # robot joint state observation
        robot_joint_state = ObsTerm(func=mdp.get_robot_body_joint_states)
        # dex3 hand joint state observation
        robot_dex3_joint_state = ObsTerm(func=mdp.get_robot_dex3_joint_states)

        def __post_init__(self):
            """post initialization function
            set the basic attributes of the observation group
            """
            self.enable_corruption = False  # disable observation value corruption
            self.concatenate_terms = False  # disable observation item connection

    @configclass
    class CameraImagesCfg(ObsGroup):
        """Observations from the robot's cameras."""

        front_camera = ObsTerm(
            func=CAMERA_IMAGE_OBS_FUNC,
            params={"sensor_cfg": SceneEntityCfg("front_camera"), "data_type": "rgb", "normalize": False},
        )
        left_wrist_camera = ObsTerm(
            func=CAMERA_IMAGE_OBS_FUNC,
            params={"sensor_cfg": SceneEntityCfg("left_wrist_camera"), "data_type": "rgb", "normalize": False},
        )
        right_wrist_camera = ObsTerm(
            func=CAMERA_IMAGE_OBS_FUNC,
            params={"sensor_cfg": SceneEntityCfg("right_wrist_camera"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.concatenate_terms = False

    # observation groups
    # create policy observation group instance
    policy: PolicyCfg = PolicyCfg()
    camera_images: CameraImagesCfg = CameraImagesCfg()


@configclass
class TerminationsCfg:
    """Termination conditions for the environment."""

    # Time out termination
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Task success termination (all stages completed)
    task_success = DoneTerm(
        func=mdp.task_success_termination,
        time_out=False,  # This is a success termination, not a failure
        params={
            "print_log": False,
            "success_stage": 2, # 1,2,3,4 for starting training, 4 for playing evaluation
        },
    )
    object_drop = DoneTerm(
        func=mdp.object_drop_termination,
        time_out=True,  # Treat as timeout/failure
        params={
            "drop_height_threshold": 0.5,  # Objects below this Z height are considered dropped
            "asset_cfg1": SceneEntityCfg("trocar_1"),
            "asset_cfg2": SceneEntityCfg("trocar_2"),
        },
    )


@configclass
class RewardsCfg:
    """Reward configuration for sparse reward mode.

    Each stage gives 1.0 reward on completion → Total reward for full task = 4.0
    This ensures clear reward signal for each stage transition.
    """

    # Stage 0: Lift trocars
    lift_trocars = RewTerm(
        func=mdp.lift_trocars_reward,
        weight=1.0,  # Give 1.0 reward when stage 0->1 completes
        params={
            "table_height": 0.85483,
            "lift_threshold": 0.15,
            "asset_cfg1": SceneEntityCfg("trocar_1"),
            "asset_cfg2": SceneEntityCfg("trocar_2"),
            # Stage transition thresholds
            "tip_align_threshold": 0.015,  # Threshold for tip alignment (m)
            "insertion_dist_threshold": 0.05,
            "insertion_angle_threshold": 0.15,
            "placement_x_min": -1.8,
            "placement_x_max": -1.4,
            "placement_y_min": 1.5,
            "placement_y_max": 1.8,
            "use_sparse_reward": True,
            "print_log": False,
        },
    )

    # Stage 1: Tip alignment (find hole)
    tip_alignment = RewTerm(
        func=mdp.trocar_tip_alignment_reward,
        weight=1.0,  # Give 1.0 reward when stage 1->2 completes
        params={
            "tip_dist_std": 0.02,  # Std for tip distance reward shaping
            "asset_cfg1": SceneEntityCfg("trocar_1"),
            "asset_cfg2": SceneEntityCfg("trocar_2"),
            "use_sparse_reward": True,
            "print_log": False,
        },
    )

    # Stage 2: Insertion (push in)
    insert_trocars = RewTerm(
        func=mdp.trocar_insertion_reward,
        weight=1.0,  # Give 1.0 reward when stage 2->3 completes
        params={
            "angle_std": 0.2,  # Std for angle alignment reward
            "angle_threshold": 0.10,  # ~5.7 degrees tolerance for parallelism
            "center_dist_std": 0.05,  # Std for center distance reward
            "asset_cfg1": SceneEntityCfg("trocar_1"),
            "asset_cfg2": SceneEntityCfg("trocar_2"),
            "use_sparse_reward": True,
            "print_log": False,
        },
    )

    # Stage 3: Placement (place in tray)
    placement_trocars = RewTerm(
        func=mdp.trocar_placement_reward,
        weight=1.0,  # Give 1.0 reward when stage 3->4 completes
        params={
            "x_min": -1.8,
            "x_max": -1.4,
            "y_min": 1.5,
            "y_max": 1.8,
            "asset_cfg1": SceneEntityCfg("trocar_1"),
            "asset_cfg2": SceneEntityCfg("trocar_2"),
            "use_sparse_reward": True,
            "print_log": False,
        },
    )


@configclass
class EventCfg:
    """Event configuration for scene reset."""

    # Reset scene when episode terminates (timeout or success)
    reset_scene = EventTermCfg(func=base_mdp.reset_scene_to_default, mode="reset")

    # Reset task stage tracker when environment resets
    reset_task_stage = EventTermCfg(func=mdp.reset_task_stage, mode="reset")

    # Random rotation for tray and trocars
    reset_tray_random_rotation = EventTermCfg(
        func=mdp.reset_tray_with_random_rotation,
        mode="reset",
        params={
            "tray_cfg": SceneEntityCfg("tray"),
            "trocar_1_cfg": SceneEntityCfg("trocar_1"),
            "trocar_2_cfg": SceneEntityCfg("trocar_2"),
            "rotation_range": [0, 10],
        },
    )

    if DEFAULT_SCENE_RANDOMIZE_LIGHTING:
        randomize_image_brightness = EventTermCfg(
            func=mdp.randomize_image_brightness,
            mode="reset",
            params={
                "brightness_range": DEFAULT_RANDOM_IMAGE_BRIGHTNESS_RANGE,
                "stratify_targets": True,
                "shuffle_targets": True,
                "print_log": DEFAULT_RANDOM_IMAGE_BRIGHTNESS_PRINT_LOG,
            },
        )


@configclass
class G1AssembleTrocarEnvCfg(ManagerBasedRLEnvCfg):
    """Unitree G1 robot assemble trocar environment configuration class
    inherits from ManagerBasedRLEnvCfg, defines all configuration parameters for the entire environment
    """

    # scene settings
    scene: InteractiveSceneCfg = ACTIVE_SCENE_CFG_CLS(
        num_envs=1,
        env_spacing=SCENE_ENV_SPACING,
        replicate_physics=SCENE_REPLICATE_PHYSICS,
    )
    # viewer settings
    viewer: ViewerCfg = ViewerCfg(
        eye=(-0.5, 2.4, 1.6),
        lookat=(-5.4, 0.2, -1.2),
        cam_prim_path="/OmniverseKit_Persp",
    )
    # basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(bounce_threshold_velocity=0.01)
        self.sim.render.enable_translucency = True
        self.sim.render.carb_settings = {
            "rtx.raytracing.fractionalCutoutOpacity": True,
        }
        self.sim.render.rendering_mode = "quality"
        self.sim.render.antialiasing_mode = "DLAA"


@configclass
class EventCfgFixTrayRotation(EventCfg):
    """Event configuration with a deterministic-but-different yaw per env index.

    This is useful for eval with many parallel envs:
      - env 0..N-1 get different yaw angles,
      - for a fixed global seed, the set of N angles is reproducible across runs/resets.

    Notes:
        - Determinism is tied to torch's global seed (set by env reset seed in IsaacLab).
        - Angle unit is degrees.
    """

    reset_tray_random_rotation = EventTermCfg(
        func=mdp.reset_tray_with_random_rotation,
        mode="reset",
        params={
            "tray_cfg": SceneEntityCfg("tray"),
            "trocar_1_cfg": SceneEntityCfg("trocar_1"),
            "trocar_2_cfg": SceneEntityCfg("trocar_2"),
            "rotation_range": [0, 10],
            "deterministic_per_env": True,
            # Use torch.initial_seed() by default to follow the env reset seed.
            "deterministic_seed": None,
        },
    )


@configclass
class G1AssembleTrocarEvalEnvCfg(G1AssembleTrocarEnvCfg):
    """Eval-friendly env cfg.

    This is currently an alias of `G1AssembleTrocarEnvCfg`, but registered under a
    separate Gym id for compatibility with RLinf configs.
    """

    # Override events to enforce deterministic per-env tray yaw on every reset.
    events: EventCfgFixTrayRotation = EventCfgFixTrayRotation()


##
# Multi-modal variant: RGB + depth + segmentation.
#
# This variant is intended for offline inspection, dataset generation, and sim-to-real
# ablations — it is NOT consumed by the RLinf training pipeline, which still pulls only
# RGB from the original `G1AssembleTrocarEnvCfg`.
##

MULTIMODAL_DATA_TYPES = (
    "rgb",
    "distance_to_image_plane",
    "semantic_segmentation",
)

# Scene-level labels are intentionally omitted from the segmentation mask path.
# The semantic filter keeps only robot, tray, and the two trocar foreground
# assets so built-in scene USD labels do not add noise to the masks.
SCENE_SUBPRIM_SEMANTIC_MAP: dict[str, str] = {}


@clone
def _spawn_scene_usd_with_subprim_labels(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Custom USD spawner that also tags sub-prims inside the spawned USD.

    Replaces ``spawn_from_usd`` as the scene's spawn function. Runs before sensor
    annotators bind, so the added ``SemanticsLabelsAPI`` labels show up in
    ``idToLabels`` for the semantic_segmentation annotator.
    """
    prim = _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)
    stage = get_current_stage()
    for child_name, class_label in SCENE_SUBPRIM_SEMANTIC_MAP.items():
        child = stage.GetPrimAtPath(f"{prim_path}/{child_name}")
        if child.IsValid():
            add_labels(child, labels=[class_label], instance_name="class", overwrite=True)
    return prim


@configclass
class AssembleTrocarSceneMultiModalCfg(ACTIVE_SCENE_CFG_CLS):
    """Scene cfg whose cameras emit RGB, depth, and segmentation.

    Also tags the task-relevant props with semantic/instance labels so the
    segmentation outputs are non-trivial (the default USD assets ship untagged).
    """

    front_camera = CameraPresets.g1_front_camera(
        data_types=list(MULTIMODAL_DATA_TYPES),
        colorize_semantic_segmentation=False,
    )
    left_wrist_camera = CameraPresets.left_dex3_wrist_camera(
        data_types=list(MULTIMODAL_DATA_TYPES),
        colorize_semantic_segmentation=False,
    )
    right_wrist_camera = CameraPresets.right_dex3_wrist_camera(
        data_types=list(MULTIMODAL_DATA_TYPES),
        colorize_semantic_segmentation=False,
    )

    def __post_init__(self):
        parent_post_init = getattr(super(), "__post_init__", None)
        if callable(parent_post_init):
            parent_post_init()
        # Attach semantic tags to task-relevant props and the robot.
        self.trocar_1.spawn.semantic_tags = [("class", "trocar")]
        self.trocar_2.spawn.semantic_tags = [("class", "trocar_device")]
        self.tray.spawn.semantic_tags = [("class", "tray")]
        if getattr(self.robot.spawn, "semantic_tags", None) is None:
            self.robot.spawn.semantic_tags = [("class", "robot")]
        # The scene USD packs multiple objects (cart, floor, trolley). We swap its
        # spawner for one that also applies per-sub-prim SemanticsLabelsAPI labels
        # on the prototype prim, so each object shows up as its own class rather
        # than inheriting a single "background" tag.
        self.scene.spawn.func = _spawn_scene_usd_with_subprim_labels


@configclass
class ObservationsMultiModalCfg(ObservationsCfg):
    """Observation groups augmented with depth and segmentation per camera."""

    @configclass
    class CameraDepthCfg(ObsGroup):
        """Per-camera depth (distance to image plane) [m]."""

        front_camera = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "distance_to_image_plane",
                "normalize": True,
            },
        )
        left_wrist_camera = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("left_wrist_camera"),
                "data_type": "distance_to_image_plane",
                "normalize": True,
            },
        )
        right_wrist_camera = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("right_wrist_camera"),
                "data_type": "distance_to_image_plane",
                "normalize": True,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = False

    @configclass
    class CameraSemanticSegmentationCfg(ObsGroup):
        """Per-camera semantic segmentation (RGBA uint8 when colorized)."""

        front_camera = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "semantic_segmentation",
                "normalize": False,
            },
        )
        left_wrist_camera = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("left_wrist_camera"),
                "data_type": "semantic_segmentation",
                "normalize": False,
            },
        )
        right_wrist_camera = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("right_wrist_camera"),
                "data_type": "semantic_segmentation",
                "normalize": False,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = False

    camera_depth: CameraDepthCfg = CameraDepthCfg()
    camera_semantic_segmentation: CameraSemanticSegmentationCfg = CameraSemanticSegmentationCfg()


@configclass
class G1AssembleTrocarMultiModalEnvCfg(G1AssembleTrocarEnvCfg):
    """Env cfg that exposes RGB, depth, and segmentation from all three cameras."""

    scene: AssembleTrocarSceneMultiModalCfg = AssembleTrocarSceneMultiModalCfg(
        num_envs=1,
        env_spacing=SCENE_ENV_SPACING,
        replicate_physics=SCENE_REPLICATE_PHYSICS,
    )
    observations: ObservationsMultiModalCfg = ObservationsMultiModalCfg()
