# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IsaacLab-owned RLinf evaluation workers.

The classes in this module intentionally live in the IsaacLab repository rather
than patching the installed RLinf package. They are opt-in from ``play.py`` and
only affect evaluation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.envs.action_utils import prepare_actions
from rlinf.scheduler import Channel
from rlinf.workers.env.env_worker import EnvWorker


def _episode_from_info(info: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return episode metrics from an RLinf info dict if present."""
    if not isinstance(info, dict):
        return None

    final_info = info.get("final_info")
    if isinstance(final_info, dict) and isinstance(final_info.get("episode"), dict):
        return final_info["episode"]

    episode = info.get("episode")
    if isinstance(episode, dict):
        return episode
    return None


def _slice_metric(value: Any, mask: torch.Tensor) -> torch.Tensor:
    """Select a metric value with a boolean env mask and return it on CPU."""
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, device=mask.device)
    elif value.device != mask.device:
        value = value.to(mask.device)
    return value[mask].detach().cpu()


def _append_episode_metrics(
    env_info: dict[str, list[torch.Tensor]],
    episode: dict[str, Any] | None,
    mask: torch.Tensor,
) -> None:
    """Append episode metrics for the selected envs."""
    if episode is None or not mask.any():
        return

    for key, value in episode.items():
        selected = _slice_metric(value, mask)
        if selected.numel() > 0:
            env_info.setdefault(key, []).append(selected)


class IsaacLabOneEpisodeEvalWorker(EnvWorker):
    """Eval worker that counts each original eval environment at most once.

    RLinf's default evaluation loop runs for a fixed rollout window. With
    ``auto_reset=True``, an environment that succeeds early can reset and finish
    a second trajectory inside the same eval window, which makes metrics such as
    ``eval/task_success_at_stage`` trajectory-level rather than one-shot
    per-environment metrics. This worker is opt-in and only changes evaluation:

    * metrics are gathered only for newly completed envs;
    * completed envs are action-masked so they do not start another attempt;
    * once all envs in a stage are done, the last observation is replayed until
      RLinf's fixed rollout loop finishes.
    """

    def _one_episode_per_env_enabled(self) -> bool:
        return bool(self.cfg.env.eval.get("one_episode_per_env", False))

    def _freeze_done_envs_enabled(self) -> bool:
        return bool(self.cfg.env.eval.get("freeze_done_envs", True))

    def _reset_one_episode_state(self) -> None:
        self._eval_finished_masks: list[torch.Tensor] = []
        self._eval_last_outputs: list[EnvOutput | None] = []

    def _finished_mask_for(self, stage_id: int, device: torch.device) -> torch.Tensor:
        mask = self._eval_finished_masks[stage_id]
        if mask.device != device:
            mask = mask.to(device)
            self._eval_finished_masks[stage_id] = mask
        return mask

    def env_evaluate_step(self, raw_actions: torch.Tensor, stage_id: int) -> tuple[EnvOutput, dict[str, Any]]:
        if not self._one_episode_per_env_enabled():
            return super().env_evaluate_step(raw_actions, stage_id)

        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.cfg.env.train.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
        )

        finished_mask = self._finished_mask_for(stage_id, chunk_actions.device)
        if finished_mask.all():
            last_output = self._eval_last_outputs[stage_id]
            if last_output is None:
                raise RuntimeError("One-episode eval state is missing the last observation.")
            return last_output, {}

        if self._freeze_done_envs_enabled() and finished_mask.any():
            chunk_actions = chunk_actions.clone()
            chunk_actions[finished_mask] = 0

        obs_list, _, chunk_terminations, chunk_truncations, infos_list = self.eval_env_list[stage_id].chunk_step(
            chunk_actions
        )
        extracted_obs = obs_list[-1] if isinstance(obs_list, (list, tuple)) and obs_list else None
        infos = infos_list[-1] if isinstance(infos_list, (list, tuple)) and infos_list else None
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)

        env_info_lists: dict[str, list[torch.Tensor]] = defaultdict(list)
        updated_finished_mask = finished_mask.clone()
        num_chunk_steps = chunk_dones.shape[1]
        for time_idx in range(num_chunk_steps):
            done_now = chunk_dones[:, time_idx].bool().to(updated_finished_mask.device)
            new_done = done_now & ~updated_finished_mask
            if not new_done.any():
                continue

            info_at_time = infos
            if isinstance(infos_list, (list, tuple)) and time_idx < len(infos_list):
                info_at_time = infos_list[time_idx]
            _append_episode_metrics(env_info_lists, _episode_from_info(info_at_time), new_done)
            updated_finished_mask |= new_done

        self._eval_finished_masks[stage_id] = updated_finished_mask
        env_info = {key: torch.cat(value, dim=0).contiguous() for key, value in env_info_lists.items() if value}

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=infos["final_observation"] if isinstance(infos, dict) and "final_observation" in infos else None,
        )
        self._eval_last_outputs[stage_id] = env_output
        return env_output, env_info

    def evaluate(self, input_channel: Channel, output_channel: Channel):
        if not self._one_episode_per_env_enabled():
            return super().evaluate(input_channel, output_channel)

        eval_metrics = defaultdict(list)
        n_chunk_steps = self.cfg.env.eval.max_steps_per_rollout_epoch // self.cfg.actor.model.num_action_chunks

        for _ in range(self.cfg.algorithm.eval_rollout_epoch):
            self._reset_one_episode_state()
            for stage_id in range(self.stage_num):
                self.eval_env_list[stage_id].is_start = True
                extracted_obs, infos = self.eval_env_list[stage_id].reset()
                env_output = EnvOutput(
                    obs=extracted_obs,
                    final_obs=infos.get("final_observation", None),
                )
                device = getattr(self.eval_env_list[stage_id], "device", torch.device("cpu"))
                self._eval_finished_masks.append(
                    torch.zeros((self.eval_num_envs_per_stage,), dtype=torch.bool, device=device)
                )
                self._eval_last_outputs.append(env_output)
                self.send_env_batch(output_channel, env_output.to_dict(), mode="eval")

            for eval_step in range(n_chunk_steps):
                for stage_id in range(self.stage_num):
                    raw_chunk_actions = self.recv_chunk_actions(input_channel, mode="eval")
                    env_output, env_info = self.env_evaluate_step(raw_chunk_actions, stage_id)

                    for key, value in env_info.items():
                        eval_metrics[key].append(value)
                    if eval_step == n_chunk_steps - 1:
                        continue
                    self.send_env_batch(output_channel, env_output.to_dict(), mode="eval")

            self.finish_rollout(mode="eval")

        for stage_id in range(self.stage_num):
            if self.cfg.env.eval.get("enable_offload", False) and hasattr(self.eval_env_list[stage_id], "offload"):
                self.eval_env_list[stage_id].offload()

        for key, value in eval_metrics.items():
            eval_metrics[key] = torch.cat(value, dim=0).contiguous().cpu()

        return eval_metrics
