#!/usr/bin/env bash

RAY_ADDRESS=local RAY_TMPDIR=/tmp/yh_original \
ASSEMBLE_TROCAR_SCENE_VARIANT=default \
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps_new_trocar_460samples \
  --num_envs 128 \
  --video \
  2>&1 | tee play_128env_450steps_yiheng_rlinf_default1.log

RAY_ADDRESS=local RAY_TMPDIR=/tmp/yh_original \
ASSEMBLE_TROCAR_SCENE_VARIANT=default \
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps_new_trocar_460samples \
  --num_envs 128 \
  --video \
  2>&1 | tee play_128env_450steps_yiheng_rlinf_default2.log

RAY_ADDRESS=local RAY_TMPDIR=/tmp/yh_original_50 \
ASSEMBLE_TROCAR_SCENE_VARIANT=default \
ASSEMBLE_TROCAR_REPLICATE_PHYSICS=false \
ASSEMBLE_TROCAR_ENV_SPACING=50.0 \
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps_new_trocar_460samples \
  --num_envs 128 \
  --video \
  2>&1 | tee play_128env_450steps_yiheng_rlinf_default3.log

RAY_ADDRESS=local RAY_TMPDIR=/tmp/yh_original \
ASSEMBLE_TROCAR_SCENE_VARIANT=default \
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps_new_trocar_460samples \
  --num_envs 128 \
  --video \
  2>&1 | tee play_128env_450steps_yiheng_rlinf_default4.log

#----------------------------------------------------------

RAY_ADDRESS=local RAY_TMPDIR=/tmp/yh_orca1 \
ASSEMBLE_TROCAR_SCENE_VARIANT=orca \
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps_new_trocar_460samples \
  --num_envs 128 \
  --video \
  2>&1 | tee play_128env_450steps_yiheng_rlinf_orca1.log

#----------------------------------------------------------

RAY_ADDRESS=local RAY_TMPDIR=/tmp/yh_factory1 \
ASSEMBLE_TROCAR_SCENE_VARIANT=factory \
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps_new_trocar_460samples \
  --num_envs 128 \
  --video \
  2>&1 | tee play_128env_450steps_yiheng_rlinf_factory1.log

#----------------------------------------------------------

RAY_ADDRESS=local RAY_TMPDIR=/tmp/yh_surgical1 \
ASSEMBLE_TROCAR_SCENE_VARIANT=surgical \
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps_new_trocar_460samples \
  --num_envs 128 \
  --video \
  2>&1 | tee play_128env_450steps_yiheng_rlinf_surgical1.log

#----------------------------------------------------------