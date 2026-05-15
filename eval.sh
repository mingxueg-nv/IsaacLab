#!/usr/bin/env bash

# RAY_ADDRESS=local RAY_TMPDIR=/tmp/mx_original \
# ASSEMBLE_TROCAR_SCENE_VARIANT=default \
# python scripts/reinforcement_learning/rlinf/play.py \
#   --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
#   --config_name isaaclab_ppo_gr00t_assemble_trocar \
#   --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/orca-dev-test/rlinf/actor/model_state_dict \
#   --num_envs 128 \
#   --video \
#   2>&1 | tee play_128env_450steps_mingxue_rlinf_default1.log

RAY_ADDRESS=local RAY_TMPDIR=/tmp/play_default1 \
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar_eval \
  --num_envs 128 \
  --video \
  2>&1 | tee play_128env_450steps_global_step_216_nodeformable_default1.log

#----------------------------------------------------------

RAY_ADDRESS=local RAY_TMPDIR=/tmp/mx_orca1 \
ASSEMBLE_TROCAR_SCENE_VARIANT=orca \
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/orca-dev-test/rlinf/actor/model_state_dict \
  --num_envs 128 \
  --video \
  2>&1 | tee play_128env_450steps_mingxue_rlinf_orca1.log

RAY_ADDRESS=local RAY_TMPDIR=/tmp/play_orca1 \
ASSEMBLE_TROCAR_SCENE_VARIANT=orca \
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar_eval \
  --num_envs 128 \
  --video \
  2>&1 | tee play_128env_450steps_global_step_216_nodeformable_orca1.log

#----------------------------------------------------------

RAY_ADDRESS=local RAY_TMPDIR=/tmp/mx_factory1 \
ASSEMBLE_TROCAR_SCENE_VARIANT=factory \
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/orca-dev-test/rlinf/actor/model_state_dict \
  --num_envs 128 \
  --video \
  2>&1 | tee play_128env_450steps_mingxue_rlinf_factory1.log

RAY_ADDRESS=local RAY_TMPDIR=/tmp/play_factory1 \
ASSEMBLE_TROCAR_SCENE_VARIANT=factory \
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar_eval \
  --num_envs 128 \
  --video \
  2>&1 | tee play_128env_450steps_global_step_216_nodeformable_factory1.log

#----------------------------------------------------------

RAY_ADDRESS=local RAY_TMPDIR=/tmp/mx_surgical1 \
ASSEMBLE_TROCAR_SCENE_VARIANT=surgical \
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/orca-dev-test/rlinf/actor/model_state_dict \
  --num_envs 128 \
  --video \
  2>&1 | tee play_128env_450steps_mingxue_rlinf_surgical1.log

RAY_ADDRESS=local RAY_TMPDIR=/tmp/play_surgical1 \
ASSEMBLE_TROCAR_SCENE_VARIANT=surgical \
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar_eval \
  --num_envs 128 \
  --video \
  2>&1 | tee play_128env_450steps_global_step_216_nodeformable_surgical1.log

#----------------------------------------------------------