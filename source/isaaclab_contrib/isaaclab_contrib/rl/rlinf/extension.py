# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RLinf extension module for IsaacLab tasks.

This module is loaded by RLinf's Worker._load_user_extensions() when
RLINF_EXT_MODULE=isaaclab_contrib.rl.rlinf.extension is set in the environment.

It registers IsaacLab tasks into RLinf's registries, allowing IsaacLab users
to train on their tasks without modifying RLinf source code.

Configuration is read from the Hydra YAML config under `env.train.isaaclab`:
    env:
      train:
        isaaclab: &isaaclab_config  # YAML anchor for reuse
          task_description: "..."
          main_images: "front_camera"
          extra_view_images: ["left_wrist_camera", "right_wrist_camera"]
          states:
            - key: "robot_joint_state"
              slice: [15, 29]
          gr00t_mapping:
            video:
              main_images: "video.room_view"
              ...
          action_mapping:
            prefix_pad: 15
      eval:
        isaaclab: *isaaclab_config  # Reuse via YAML anchor

Task IDs are read automatically from ``env.train.init_params.id`` and
``env.eval.init_params.id`` in the YAML config.

Usage:
    export RLINF_EXT_MODULE=isaaclab_contrib.rl.rlinf.extension
    export RLINF_CONFIG_FILE=/path/to/isaaclab_ppo_gr00t_assemble_trocar.yaml
"""

from __future__ import annotations

import collections.abc
import logging
import os
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

_registered = False

# Cache for YAML config (loaded once per process)
_full_cfg_cache: dict | None = None

# one-shot init-pose debug flag (TEMP: remove after verification)
_pose_logged: bool = False


def register() -> None:
    """Register IsaacLab extensions into RLinf.

    This function is called automatically by RLinf's Worker._load_user_extensions()
    when RLINF_EXT_MODULE=isaaclab_contrib.rl.rlinf.extension is set.

    It performs the following registrations:
    1. Registers GR00T obs/action converters
    2. Registers GR00T data config
    3. Patches GR00T get_model for custom embodiment
    4. Registers task IDs from YAML config (env.*.init_params.id) into REGISTER_ISAACLAB_ENVS
    """
    global _registered
    if _registered:
        return
    _registered = True

    logger.info("isaaclab_contrib.rl.rlinf.extension: Registering IsaacLab extensions...")

    # Load config once and pass to all registration functions
    cfg = _get_isaaclab_cfg()

    _register_gr00t_converters(cfg)
    _patch_gr00t_get_model(cfg)
    _register_isaaclab_envs()

    logger.info("isaaclab_contrib.rl.rlinf.extension: Registration complete.")


def _load_full_cfg() -> dict:
    """Load and cache the full YAML config from ``RLINF_CONFIG_FILE``.

    Raises:
        ValueError: If the ``RLINF_CONFIG_FILE`` environment variable is not set.

    Returns:
        The parsed YAML config as a nested dictionary.
    """
    global _full_cfg_cache
    if _full_cfg_cache is not None:
        return _full_cfg_cache
    config_file = os.environ.get("RLINF_CONFIG_FILE", "")
    if not config_file:
        raise ValueError("RLINF_CONFIG_FILE not set")
    with open(config_file) as f:
        _full_cfg_cache = yaml.safe_load(f)
    logger.info(f"Loaded full config from {config_file}")
    return _full_cfg_cache


def _get_isaaclab_cfg() -> dict:
    """Return the ``env.train.isaaclab`` section from the cached full config.

    Returns:
        The IsaacLab-specific configuration dictionary. Empty dict if the section is missing.
    """
    return _load_full_cfg().get("env", {}).get("train", {}).get("isaaclab", {})


def _get_model_type() -> str:
    """Return ``actor.model.model_type`` from the cached YAML config.

    Falls back to ``"gr00t"`` (N1.5) for backwards compatibility with configs
    that predate the N1.7 split.
    """
    return (
        _load_full_cfg()
        .get("actor", {})
        .get("model", {})
        .get("model_type", "gr00t")
    )


def _is_gr00t_n17(model_type: str | None = None) -> bool:
    """Return True if the active config is for GR00T N1.7 (a.k.a. ``gr00t_1_7``)."""
    if model_type is None:
        model_type = _get_model_type()
    return model_type in ("gr00t_1_7", "gr00t_17", "gr00t_n1d7")


def _get_embodiment_tags_module():
    """Return the RLinf ``embodiment_tags`` module that matches ``model_type``.

    In the current RLinf codebase, embodiment_tags lives under
    ``rlinf.models.embodiment.gr00t.embodiment_tags`` for all model versions
    (N1.5, N1.6, N1.7). The gr00t_n1d7 get_model patches
    ``gr00t.data.embodiment_tags`` from this same module.
    """
    from rlinf.models.embodiment.gr00t import embodiment_tags as _et
    return _et


def _get_simulation_io_module():
    """Return the RLinf ``simulation_io`` module that matches ``model_type``.

    In the current RLinf codebase, simulation_io lives under
    ``rlinf.models.embodiment.gr00t.simulation_io`` for all model versions.
    """
    from rlinf.models.embodiment.gr00t import simulation_io as _sio
    return _sio


def _patch_embodiment_tags(cfg: dict) -> None:
    """Add custom embodiment tag to RLinf's EmbodimentTag enum and mapping if needed.

    Reads ``embodiment_tag`` and ``embodiment_tag_id`` from the IsaacLab config section.
    Only adds the tag if it is not already present in RLinf's native registry.

    Args:
        cfg: The IsaacLab-specific configuration dictionary (``env.train.isaaclab``).
    """
    # GR00T uses embodiment tags to identify different robots.  Custom robots
    # (like G129+Dex3) need a unique tag string and numeric ID so that the
    # model's tokenizer can map them to the correct action/state dimensions.
    #
    # The numeric ID is the projector index in GR00T's Action Expert Module.
    # Known mapping (from gr00t/data/embodiment_tags.py):
    #   17 = oxe_droid, 24 = gr1, 26 = agibot_genie1, 31 = new_embodiment
    # Default 31 corresponds to the "new_embodiment" slot reserved for
    # fine-tuning on custom robots.
    embodiment_tag = cfg.get("embodiment_tag", "new_embodiment")
    tag_id = cfg.get("embodiment_tag_id", 31)

    # Pick the right embodiment_tags module for the active model_type. N1.5
    # and N1.7 maintain independent copies; patching the wrong one silently
    # fails (the model's tokenizer would then map the custom tag to the
    # default projector slot or raise at load time).
    embodiment_tags = _get_embodiment_tags_module()

    # If tag is already in registry (native or previously added), skip
    if embodiment_tag in embodiment_tags.EMBODIMENT_TAG_MAPPING:
        logger.info(f"embodiment_tag '{embodiment_tag}' already registered")
        return
    # Add to enum
    tag_upper = embodiment_tag.upper().replace("-", "_")
    if not hasattr(embodiment_tags.EmbodimentTag, tag_upper):
        existing_members = {e.name: e.value for e in embodiment_tags.EmbodimentTag}
        existing_members[tag_upper] = embodiment_tag
        NewEmbodimentTag = Enum("EmbodimentTag", existing_members)

        embodiment_tags.EmbodimentTag = NewEmbodimentTag
        logger.info(f"Added EmbodimentTag.{tag_upper} = '{embodiment_tag}'")

    # Add to mapping
    embodiment_tags.EMBODIMENT_TAG_MAPPING[embodiment_tag] = tag_id
    logger.info(f"Added EMBODIMENT_TAG_MAPPING['{embodiment_tag}'] = {tag_id}")


def _patch_gr00t_get_model(cfg: dict) -> None:
    """Monkeypatch RLinf's GR00T ``get_model`` to support custom ``data_config``.

    The patch is applied only when the user specifies a ``data_config_class`` in the
    YAML config. Embodiment tags are always ensured to be registered.

    Args:
        cfg: The IsaacLab-specific configuration dictionary (``env.train.isaaclab``).
    """
    # Always ensure embodiment tag is registered
    _patch_embodiment_tags(cfg)
    # Only patch get_model if user wants custom data_config
    data_config_class = cfg.get("data_config_class", "")
    if not data_config_class:
        logger.info("No data_config_class specified, using RLinf's default get_model")
        return

    # N1.7 removed gr00t.experiment.data_config (and BaseDataConfig /
    # load_data_config / DATA_CONFIG_MAP with it). The monkeypatch below is
    # hard-wired to the N1.5 GR00T_N1_5_ForRLActionPrediction path and cannot
    # apply against N1.7. If a user accidentally leaves data_config_class set
    # in an N1.7 yaml, refuse loudly rather than silently importing N1.5 code.
    if _is_gr00t_n17():
        raise RuntimeError(
            "data_config_class is set in an N1.7 (gr00t_1_7) config; this is a "
            "legacy N1.5-only knob. Remove data_config_class from the YAML — "
            "the N1.7 adapter loads modality_config straight from the ckpt's "
            "experiment_cfg/config.yaml."
        )

    import rlinf.models.embodiment.gr00t as rlinf_gr00t_mod

    def patched_get_model(model_cfg, torch_dtype=None) -> object:
        """Load a GR00T model with custom ``data_config`` and embodiment tag.

        Args:
            model_cfg: RLinf model configuration object containing ``model_path``,
                ``embodiment_tag``, ``denoising_steps``, ``num_action_chunks``,
                ``obs_converter_type``, and ``rl_head_config``.
            torch_dtype: The torch dtype for the model. Defaults to ``torch.bfloat16``.

        Raises:
            FileNotFoundError: If ``model_cfg.model_path`` does not exist.

        Returns:
            The loaded GR00T model instance.
        """
        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        # Handle custom embodiment (we only get here if tag was not natively supported)
        from gr00t.experiment.data_config import load_data_config
        from rlinf.models.embodiment.gr00t.gr00t_action_model import GR00T_N1_5_ForRLActionPrediction
        from rlinf.models.embodiment.gr00t.utils import replace_dropout_with_identity
        from rlinf.utils.patcher import Patcher

        # Apply RLinf's standard EmbodimentTag patches
        Patcher.clear()
        Patcher.add_patch(
            "gr00t.data.embodiment_tags.EmbodimentTag",
            "rlinf.models.embodiment.gr00t.embodiment_tags.EmbodimentTag",
        )
        Patcher.add_patch(
            "gr00t.data.embodiment_tags.EMBODIMENT_TAG_MAPPING",
            "rlinf.models.embodiment.gr00t.embodiment_tags.EMBODIMENT_TAG_MAPPING",
        )
        Patcher.apply()

        data_config = load_data_config(data_config_class)
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        model_path = Path(model_cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # rl_model_path: optional path to an RLinf checkpoint with full_weights.pt
        rl_model_path = getattr(model_cfg, "rl_model_path", None)

        model = GR00T_N1_5_ForRLActionPrediction.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            embodiment_tag=model_cfg.embodiment_tag,
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=model_cfg.denoising_steps,
            output_action_chunks=model_cfg.num_action_chunks,
            obs_converter_type=model_cfg.obs_converter_type,
            tune_visual=False,
            tune_llm=False,
            rl_head_config=model_cfg.rl_head_config,
        )

        if rl_model_path:
            rl_weights = Path(rl_model_path) / "actor" / "model_state_dict" / "full_weights.pt"
            if not rl_weights.exists():
                raise FileNotFoundError(
                    f"rl_model_path={rl_model_path}: cannot find full_weights.pt "
                    f"(tried directly and under actor/model_state_dict/)"
                )
            logger.info(f"Loading RL finetuned weights from {rl_weights}")
            state_dict = torch.load(rl_weights, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)

        model.to(torch_dtype)
        if model_cfg.rl_head_config.add_value_head:
            model.action_head.value_head._init_weights()
        if model_cfg.rl_head_config.disable_dropout:
            replace_dropout_with_identity(model)

        logger.info(f"Loaded GR00T model with embodiment_tag='{model_cfg.embodiment_tag}'")
        return model

    rlinf_gr00t_mod.get_model = patched_get_model
    logger.info(f"Patched get_model for data_config_class='{data_config_class}'")


def _register_gr00t_converters(cfg: dict) -> None:
    """Register GR00T obs/action converters for IsaacLab tasks.

    Reads ``obs_converter_type`` from the YAML config (``env.train.isaaclab.obs_converter_type``)
    and registers the corresponding observation and action conversion functions into
    RLinf's ``simulation_io`` registry that matches ``actor.model.model_type``.

    The N1.5 (``rlinf.models.embodiment.gr00t.simulation_io``) and N1.7
    (``rlinf.models.embodiment.gr00t_1_7.simulation_io``) modules each carry an
    independent ``OBS_CONVERSION`` / ``ACTION_CONVERSION`` dict. The N1.7
    action-model resolves its converter against the gr00t_1_7 module at
    ``__init__`` time, so we must register there for the N1.7 path.

    Args:
        cfg: The IsaacLab-specific configuration dictionary (``env.train.isaaclab``).
    """
    simulation_io = _get_simulation_io_module()
    # N1.5 default is "dex3"; N1.7 yamls use "passthrough" (the action mapping
    # lives in extension.py and is driven by the yaml's gr00t_mapping block).
    default_converter = "passthrough" if _is_gr00t_n17() else "dex3"
    obs_converter_type = cfg.get("obs_converter_type", default_converter)

    if obs_converter_type not in simulation_io.OBS_CONVERSION:
        simulation_io.OBS_CONVERSION[obs_converter_type] = _convert_isaaclab_obs_to_gr00t
        logger.info(
            f"Registered obs converter '{obs_converter_type}' into "
            f"{simulation_io.__name__}"
        )

    model_type = _get_model_type()
    if model_type in ("gr00t_1_7", "gr00t_17", "gr00t_n1d7"):
        action_dict = simulation_io.ACTION_CONVERSION_N1D7
    elif model_type in ("gr00t_n1d6", "gr00t_1_6"):
        action_dict = simulation_io.ACTION_CONVERSION_N1D6
    else:
        action_dict = simulation_io.ACTION_CONVERSION_N1D5

    if obs_converter_type not in action_dict:
        action_dict[obs_converter_type] = _convert_gr00t_to_isaaclab_action
        logger.info(
            f"Registered action converter '{obs_converter_type}' into "
            f"{simulation_io.__name__} (dict for {model_type})"
        )


def _convert_isaaclab_obs_to_gr00t(env_obs: dict) -> dict:
    """Convert IsaacLab env observations to GR00T format.

    Uses ``gr00t_mapping`` from the YAML config (``env.train.isaaclab.gr00t_mapping``)
    to map IsaacLab observation keys to GR00T-expected keys.

    Args:
        env_obs: Observation dictionary from ``_wrap_obs`` with the following keys:

            - ``"main_images"``: ``(B, H, W, C)`` torch tensor.
            - ``"extra_view_images"``: ``(B, N, H, W, C)`` torch tensor.
            - ``"states"``: ``(B, D)`` torch tensor.
            - ``"task_descriptions"``: list of strings.

    Returns:
        A dictionary with GR00T-formatted observations (numpy arrays with a time
        dimension, e.g. ``(B, T=1, H, W, C)``).
    """
    groot_obs = {}
    # Load mapping config from YAML or env var
    cfg = _get_isaaclab_cfg()
    gr00t_mapping = cfg.get("gr00t_mapping", {})
    video_mapping = gr00t_mapping.get("video", {})
    state_mapping = gr00t_mapping.get("state", [])
    # Convert main_images -> video.xxx
    if "main_images" in env_obs:
        main = env_obs["main_images"]
        gr00t_key = video_mapping.get("main_images", "video.room_view")
        if isinstance(main, torch.Tensor):
            # (B, H, W, C) -> (B, T=1, H, W, C)
            groot_obs[gr00t_key] = main.unsqueeze(1).cpu().numpy()
    # Convert extra_view_images -> video.xxx
    if "extra_view_images" in env_obs:
        extra = env_obs["extra_view_images"]  # (B, N, H, W, C)
        extra_keys = video_mapping.get("extra_view_images", [])
        if isinstance(extra, torch.Tensor):
            for i, key in enumerate(extra_keys):
                if i < extra.shape[1]:
                    # (B, H, W, C) -> (B, T=1, H, W, C)
                    groot_obs[key] = extra[:, i].unsqueeze(1).cpu().numpy()
    # Convert states -> state.xxx with slicing
    if "states" in env_obs and state_mapping:
        states = env_obs["states"]  # (B, D)
        if isinstance(states, torch.Tensor):
            states_np = states.unsqueeze(1).cpu().numpy()  # (B, T=1, D)
            for spec in state_mapping:
                gr00t_key = spec.get("gr00t_key")
                slice_range = spec.get("slice", [0, states_np.shape[-1]])
                if gr00t_key:
                    full_key = f"state.{gr00t_key}" if not gr00t_key.startswith("state.") else gr00t_key
                    groot_obs[full_key] = states_np[:, :, slice_range[0] : slice_range[1]]

    # Pass through task descriptions
    groot_obs["annotation.human.task_description"] = env_obs.get("task_descriptions", [])

    return groot_obs


def _convert_gr00t_to_isaaclab_action(action_chunk: dict, chunk_size: int = 1) -> np.ndarray:
    """Convert GR00T action output to IsaacLab env action format.

    Uses ``action_mapping`` from the YAML config (``env.train.isaaclab.action_mapping``)
    to apply optional prefix/suffix zero-padding to the concatenated action vector.

    Args:
        action_chunk: Dictionary of action arrays from GR00T, each with shape
            ``(B, T, D_i)``.
        chunk_size: Number of time steps to keep from the action chunk. Defaults to 1.

    Returns:
        Concatenated and padded action array with shape ``(B, chunk_size, D)``.
    """

    # Load mapping config from YAML or env var
    cfg = _get_isaaclab_cfg()
    action_mapping = cfg.get("action_mapping", {})
    prefix_pad = action_mapping.get("prefix_pad", 0)
    suffix_pad = action_mapping.get("suffix_pad", 0)

    # Concatenate all action parts
    action_parts = [v[:, :chunk_size, :] for v in action_chunk.values()]
    action_concat = np.concatenate(action_parts, axis=-1)

    # Apply padding
    if prefix_pad > 0 or suffix_pad > 0:
        action_concat = np.pad(
            action_concat,
            ((0, 0), (0, 0), (prefix_pad, suffix_pad)),
            mode="constant",
            constant_values=0,
        )
    return action_concat


def _register_isaaclab_envs() -> None:
    """Register IsaacLab tasks into RLinf's REGISTER_ISAACLAB_ENVS map.

    Task IDs are read from ``env.train.init_params.id`` and
    ``env.eval.init_params.id`` in the YAML config.
    """
    from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS

    # Collect unique task IDs from the YAML config (train + eval)
    full_cfg = _load_full_cfg()
    env_cfg = full_cfg.get("env", {})
    task_ids: list[str] = []
    for section in ("train", "eval"):
        tid = env_cfg.get(section, {}).get("init_params", {}).get("id", "")
        if tid and tid not in task_ids:
            task_ids.append(tid)

    if not task_ids:
        logger.warning("No task IDs found in YAML config (env.*.init_params.id)")
        return

    logger.info(f"Tasks to register: {task_ids}")

    for task_id in task_ids:
        if task_id in REGISTER_ISAACLAB_ENVS:
            logger.debug(f"Task '{task_id}' already registered, skipping")
            continue

        # Create a generic wrapper class for this task
        env_class = _create_generic_env_wrapper(task_id)
        REGISTER_ISAACLAB_ENVS[task_id] = env_class
        logger.info(f"Registered IsaacLab task '{task_id}' for RLinf")

    logger.debug(f"REGISTER_ISAACLAB_ENVS now contains: {list(REGISTER_ISAACLAB_ENVS.keys())}")


def _create_generic_env_wrapper(task_id: str) -> type:
    """Create a generic wrapper class for an IsaacLab task.

    The wrapper class will load the task configuration at runtime
    (after AppLauncher starts) and configure observation mapping accordingly.

    This follows the same pattern as i4h's rlinf_ext: all isaaclab-dependent
    imports happen inside _make_env_function, after AppLauncher starts.

    Args:
        task_id: The gymnasium task ID.

    Returns:
        A class that inherits from IsaaclabBaseEnv.
    """
    from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv

    _task_id = task_id

    class IsaacLabGenericEnv(IsaaclabBaseEnv):
        """Generic environment wrapper for IsaacLab tasks.

        Config is read from the YAML file via ``_get_isaaclab_cfg()``.
        """

        def __init__(self, cfg, num_envs: int, seed_offset: int, total_num_processes: int, worker_info):
            """Initialize the generic IsaacLab environment wrapper.

            Args:
                cfg: RLinf environment configuration object.
                num_envs: Number of parallel environments.
                seed_offset: Seed offset for reproducibility.
                total_num_processes: Total number of worker processes.
                worker_info: RLinf worker metadata.
            """
            super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

        def _record_metrics(self, step_reward, terminations, infos):
            """Override to use terminations (task completion) for success_once."""

            episode_info = {}
            self.returns += step_reward
            self.success_once = self.success_once | terminations.bool()
            episode_info["success_once"] = self.success_once.clone()
            episode_info["return"] = self.returns.clone()
            episode_info["episode_len"] = self.elapsed_steps.clone()
            episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
            infos["episode"] = episode_info
            return infos

        def _make_env_function(self) -> collections.abc.Callable:
            """Create the environment factory function.

            This function runs in a child process (via ``SubProcIsaacLabEnv``).
            All IsaacLab-dependent imports happen here, after ``AppLauncher`` starts.

            Returns:
                A callable that creates and returns the IsaacLab environment and sim app.
            """

            def make_env_isaaclab() -> tuple:
                """Create the IsaacLab environment inside the child process.

                Returns:
                    A tuple of ``(env, sim_app)`` where ``env`` is the unwrapped
                    gymnasium environment and ``sim_app`` is the Isaac Sim application.
                """
                from isaaclab.app import AppLauncher

                sim_app = AppLauncher(headless=True, enable_cameras=True).app
                import gymnasium as gym

                from isaaclab_tasks.utils import load_cfg_from_registry

                isaac_env_cfg = load_cfg_from_registry(self.isaaclab_env_id, "env_cfg_entry_point")
                isaac_env_cfg.scene.num_envs = self.cfg.init_params.num_envs

                env = gym.make(self.isaaclab_env_id, cfg=isaac_env_cfg, render_mode="rgb_array").unwrapped

                # Fix stale first camera frame after reset (mirrors the working
                # GR00T 1.5 extension): pump one Kit frame without advancing
                # physics, force sensors to re-read, recompute observations.
                import omni.kit.app as _kit_app_mod
                _app = _kit_app_mod.get_app()
                _original_reset = env.reset
                _diag = [0]

                def _patched_reset(*args, **kwargs):
                    obs, extras = _original_reset(*args, **kwargs)
                    try:
                        import isaaclab_physx.renderers.isaac_rtx_renderer_utils as _rtx_utils
                    except Exception:
                        _rtx_utils = None
                    # Aggressively flush the RTX render pipeline.
                    # The annotator data lags app.update() by >= 1 frame,
                    # so we pump many frames to guarantee convergence.
                    env.sim.set_setting("/app/player/playSimulations", False)
                    for _flush_i in range(10):
                        if _rtx_utils is not None:
                            _rtx_utils._last_render_update_key = (0, -1)
                        _app.update()
                    env.sim.set_setting("/app/player/playSimulations", True)
                    # Clear transform-step dedup in RenderContext so the
                    # next sensor read triggers update_transforms again.
                    _sim_ctx = env.sim
                    if hasattr(_sim_ctx, 'render_context'):
                        _rc = _sim_ctx.render_context
                        if hasattr(_rc, 'reset_transform_cadence'):
                            _rc.reset_transform_cadence()
                        if hasattr(_rc, '_last_transforms_step'):
                            _rc._last_transforms_step = None
                    # Reset annotator dedup and re-read all sensors.
                    if _rtx_utils is not None:
                        _rtx_utils._last_render_update_key = (0, -1)
                    for _sensor in env.scene.sensors.values():
                        _sensor.update(dt=0.0, force_recompute=True)
                    obs = env.observation_manager.compute(update_history=True)
                    env.obs_buf = obs
                    # Save diagnostic image from the recomputed observation
                    if _diag[0] < 1:
                        try:
                            import torch as _th
                            _cam_obs = obs.get("camera_images", {})
                            for _ck, _cv in _cam_obs.items():
                                _img_t = _cv[0] if _cv.dim() == 4 else _cv
                                _img_t = _img_t.detach().cpu()
                                if hasattr(_img_t, "torch"):
                                    _img_t = _img_t.torch
                                _np_img = _img_t.numpy()
                                import numpy as _np2
                                if _np_img.max() <= 1.0 and _np_img.dtype in (_np2.float32, _np2.float16):
                                    _np_img = (_np_img * 255).clip(0, 255).astype(_np2.uint8)
                                _ppm_path = "/tmp/diag_reset_%s.ppm" % _ck
                                _h, _w = _np_img.shape[:2]
                                _c = _np_img.shape[2] if _np_img.ndim == 3 else 1
                                with open(_ppm_path, "wb") as _pf:
                                    _pf.write(("P6\n%d %d\n255\n" % (_w, _h)).encode())
                                    if _c == 4:
                                        _pf.write(_np_img[:,:,:3].tobytes())
                                    elif _c == 3:
                                        _pf.write(_np_img.tobytes())
                                    else:
                                        _pf.write(_np2.stack([_np_img]*3, axis=-1).tobytes())
                                print("[DIAG] saved %s shape=%s" % (_ppm_path, list(_np_img.shape)), flush=True)
                        except Exception as _save_e:
                            print("[DIAG] image save failed: %s" % _save_e, flush=True)
                    if _diag[0] < 1:
                        _diag[0] += 1
                        _robot = None
                        _nj = -1
                        for _a in env.scene.articulations.values():
                            _jp = _a.data.joint_pos
                            _jp = _jp.torch if hasattr(_jp, "torch") else _jp
                            if _jp.shape[-1] > _nj:
                                _nj = _jp.shape[-1]
                                _robot = _a
                        _jp = _robot.data.joint_pos
                        _jp = _jp.torch if hasattr(_jp, "torch") else _jp
                        _dj = _robot.data.default_joint_pos
                        _dj = _dj.torch if hasattr(_dj, "torch") else _dj
                        print("[DIAG] reset joint vs default max|diff|=%.6f" % (_jp[0] - _dj[0]).abs().max().item(), flush=True)
                        for _nm, _sn in env.scene.sensors.items():
                            _p = getattr(getattr(_sn, "data", None), "pos_w", None)
                            if _p is not None:
                                _p = _p.torch if hasattr(_p, "torch") else _p
                                print("[DIAG] reset cam %s pos_w=%s" % (_nm, [round(float(x), 4) for x in _p[0].tolist()]), flush=True)
                    return obs, extras

                env.reset = _patched_reset

                _step_diag = [0]
                _original_step = env.step

                def _patched_step(*args, **kwargs):
                    out = _original_step(*args, **kwargs)
                    if _step_diag[0] < 1:
                        _step_diag[0] += 1
                        for _nm, _sn in env.scene.sensors.items():
                            _p = getattr(getattr(_sn, "data", None), "pos_w", None)
                            if _p is not None:
                                _p = _p.torch if hasattr(_p, "torch") else _p
                                print("[DIAG] step1 cam %s pos_w=%s" % (_nm, [round(float(x), 4) for x in _p[0].tolist()]), flush=True)
                    return out

                env.step = _patched_step

                return env, sim_app

            return make_env_isaaclab

        def _wrap_obs(self, obs: dict) -> dict:
            """Convert IsaacLab observations to the RLinf format.

            The output format matches i4h's convention:

            - ``"main_images"``: ``(B, H, W, C)`` — single main camera.
            - ``"extra_view_images"``: ``(B, N, H, W, C)`` — stacked extra cameras.
            - ``"states"``: ``(B, D)`` — concatenated state vector.
            - ``"task_descriptions"``: ``list[str]`` — task descriptions.
            Config is read from the YAML file via :func:`_get_isaaclab_cfg`.

            Args:
                obs: Raw observation dictionary from the IsaacLab environment.

            Returns:
                A dictionary with observations mapped to the RLinf convention.
            """
            # import torch

            policy_obs = obs.get("policy", obs)
            camera_obs = obs.get("camera_images", {})

            # TEMP one-shot init-pose check (remove after verification)
            global _pose_logged
            if not _pose_logged:
                _pose_logged = True
                try:
                    _rjs = policy_obs.get("robot_joint_state")
                    _rds = policy_obs.get("robot_dex3_joint_state")
                    if _rjs is not None:
                        logger.warning("[INIT-POSE-CHECK] robot_joint_state[0]=%s", _rjs[0].detach().cpu().tolist())
                    if _rds is not None:
                        logger.warning("[INIT-POSE-CHECK] robot_dex3_joint_state[0]=%s", _rds[0].detach().cpu().tolist())
                except Exception as _e:
                    logger.warning("[INIT-POSE-CHECK] failed: %s", _e)

            cfg = _get_isaaclab_cfg()
            # Get task description from config
            task_desc = cfg.get("task_description", "") or self.task_description
            rlinf_obs = {
                "task_descriptions": [task_desc] * self.num_envs,
            }

            if not cfg:
                logger.warning("IsaacLab config is empty, returning minimal observation")
                return rlinf_obs

            # main_images: single camera key -> (B, H, W, C)
            main_key = cfg.get("main_images")
            if main_key and main_key in camera_obs:
                rlinf_obs["main_images"] = camera_obs[main_key]

            # extra_view_images: camera key(s) -> stack to (B, N, H, W, C)
            extra_keys = cfg.get("extra_view_images")
            if extra_keys:
                if isinstance(extra_keys, str):
                    extra_keys = [extra_keys]
                extra_imgs = [camera_obs[k] for k in extra_keys if k in camera_obs]
                if extra_imgs:
                    rlinf_obs["extra_view_images"] = torch.stack(extra_imgs, dim=1)

            # states: list of state specs -> concatenate to (B, D)
            # Each spec: string "key" or dict {"key": "...", "slice": [start, end]}
            state_specs = cfg.get("states")
            if state_specs:
                state_parts = []
                for spec in state_specs:
                    if isinstance(spec, str):
                        state = policy_obs.get(spec)
                        if state is not None:
                            state_parts.append(state)
                    elif isinstance(spec, dict):
                        state = policy_obs.get(spec.get("key"))
                        if state is not None:
                            slice_range = spec.get("slice")
                            if slice_range:
                                state = state[:, slice_range[0] : slice_range[1]]
                            state_parts.append(state)
                if state_parts:
                    rlinf_obs["states"] = torch.cat(state_parts, dim=-1)

            return rlinf_obs

        def add_image(self, obs: dict) -> np.ndarray | None:
            """Get image for video logging.

            Args:
                obs: Raw observation dictionary from the IsaacLab environment.

            Returns:
                A numpy array of shape ``(H, W, C)`` for the first environment, or
                ``None`` if no camera image is available.
            """
            camera_obs = obs.get("camera_images", {})
            cfg = _get_isaaclab_cfg()
            # Try main_images key, fallback to first available camera
            main_key = cfg.get("main_images")
            if main_key and main_key in camera_obs:
                return camera_obs[main_key][0].cpu().numpy()
            for img in camera_obs.values():
                return img[0].cpu().numpy()
            return None

    return IsaacLabGenericEnv
