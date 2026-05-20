# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IsaacLab-owned RLinf training workers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.envs.action_utils import prepare_actions
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


class IsaacLabChunkMetricsEnvWorker(EnvWorker):
    """EnvWorker with correct episode metrics for chunked IsaacLab actions.

    RLinf's default IsaacLab ``chunk_step`` collapses a chunk's done flags to
    the last substep when ``auto_reset=True``. That is fine for rollout data,
    but episode metrics such as ``task_success_at_stage`` must be taken from
    the exact substep where the episode ended. Otherwise AC8 can report the
    metric from a post-reset episode and make task success look falsely low.
    """

    def _chunk_step_with_raw_dones(self, env, chunk_actions):
        """Mirror RLinf's chunk_step and also return raw per-substep dones."""
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []

        for time_idx in range(chunk_size):
            actions = chunk_actions[:, time_idx]
            obs, step_reward, terminations, truncations, infos = env.step(actions, auto_reset=False)
            obs_list.append(obs)
            infos_list.append(infos)
            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        raw_chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)
        raw_chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and env.auto_reset:
            obs_list[-1], infos_list[-1] = env._handle_auto_reset(past_dones, obs_list[-1], infos_list[-1])

        if env.auto_reset or env.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations).to(env.device)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations).to(env.device)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()

        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
            raw_chunk_terminations,
            raw_chunk_truncations,
        )

    def _episode_metrics_from_raw_dones(
        self,
        infos_list: list[dict[str, Any]],
        raw_chunk_dones: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Collect the first completed episode for each env inside a chunk."""
        env_info_lists: dict[str, list[torch.Tensor]] = defaultdict(list)
        already_counted = torch.zeros(raw_chunk_dones.shape[0], dtype=torch.bool, device=raw_chunk_dones.device)

        for time_idx in range(raw_chunk_dones.shape[1]):
            done_now = raw_chunk_dones[:, time_idx].bool()
            new_done = done_now & ~already_counted
            if not new_done.any():
                continue

            info_at_time = infos_list[time_idx] if time_idx < len(infos_list) else None
            _append_episode_metrics(env_info_lists, _episode_from_info(info_at_time), new_done)
            already_counted |= new_done

        return {key: torch.cat(value, dim=0).contiguous() for key, value in env_info_lists.items() if value}

    def env_interact_step(self, raw_actions: torch.Tensor, stage_id: int) -> tuple[EnvOutput, dict[str, Any]]:
        """Run one train env interaction step with AC episode metrics fixed."""
        if self.cfg.actor.model.num_action_chunks <= 1:
            return super().env_interact_step(raw_actions, stage_id)

        chunk_actions = prepare_actions(
            raw_chunk_actions=raw_actions,
            env_type=self.cfg.env.train.env_type,
            model_type=self.cfg.actor.model.model_type,
            num_action_chunks=self.cfg.actor.model.num_action_chunks,
            action_dim=self.cfg.actor.model.action_dim,
            policy=self.cfg.actor.model.get("policy_setup", None),
            wm_env_type=self.cfg.env.train.get("wm_env_type", None),
        )

        (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
            raw_chunk_terminations,
            raw_chunk_truncations,
        ) = self._chunk_step_with_raw_dones(self.env_list[stage_id], chunk_actions)

        extracted_obs = obs_list[-1] if isinstance(obs_list, (list, tuple)) and obs_list else None
        infos = infos_list[-1] if isinstance(infos_list, (list, tuple)) and infos_list else {}
        chunk_dones = torch.logical_or(chunk_terminations, chunk_truncations)
        raw_chunk_dones = torch.logical_or(raw_chunk_terminations, raw_chunk_truncations)

        if self.cfg.env.train.auto_reset or (
            not self.cfg.env.train.auto_reset and not self.cfg.env.train.ignore_terminations
        ):
            env_info = self._episode_metrics_from_raw_dones(infos_list, raw_chunk_dones)
        else:
            env_info = {}
            if chunk_truncations[:, -1].any():
                assert chunk_truncations[:, -1].all()
                if "episode" in infos:
                    for key, value in infos["episode"].items():
                        env_info[key] = value.cpu()

        intervene_actions = infos.get("intervene_action") if isinstance(infos, dict) else None
        intervene_flags = infos.get("intervene_flag") if isinstance(infos, dict) else None
        if self.cfg.env.train.auto_reset and chunk_dones.any() and isinstance(infos, dict) and "final_info" in infos:
            final_info = infos["final_info"]
            if isinstance(final_info, dict) and "intervene_action" in final_info:
                intervene_actions = final_info["intervene_action"]
                intervene_flags = final_info.get("intervene_flag")

        env_output = EnvOutput(
            obs=extracted_obs,
            final_obs=infos["final_observation"] if isinstance(infos, dict) and "final_observation" in infos else None,
            rewards=chunk_rewards,
            dones=chunk_dones,
            terminations=chunk_terminations,
            truncations=chunk_truncations,
            intervene_actions=intervene_actions,
            intervene_flags=intervene_flags,
        )
        return env_output, env_info
