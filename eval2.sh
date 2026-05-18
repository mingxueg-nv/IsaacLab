#!/usr/bin/env bash

# RAY_ADDRESS=local RAY_TMPDIR=/tmp/yh_original \
# ASSEMBLE_TROCAR_SCENE_VARIANT=default \
# python scripts/reinforcement_learning/rlinf/play.py \
#   --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
#   --config_name isaaclab_ppo_gr00t_assemble_trocar \
#   --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps_new_trocar_460samples \
#   --num_envs 128 \
#   --video \
#   2>&1 | tee play_128env_450steps_yiheng_rlinf_default2.log


# RAY_ADDRESS=local RAY_TMPDIR=/tmp/yh_original \
# ASSEMBLE_TROCAR_SCENE_VARIANT=default \
# python scripts/reinforcement_learning/rlinf/play.py \
#   --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
#   --config_name isaaclab_ppo_gr00t_assemble_trocar_ac8 \
#   --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps_new_trocar_460samples \
#   --num_envs 128 \
#   --video \
#   2>&1 | tee play_128env_450steps_yiheng_rlinf_default1_ac8.log

#----------------------------------------------------------

# RAY_ADDRESS=local RAY_TMPDIR=/tmp/yh_orca1 \
# ASSEMBLE_TROCAR_SCENE_VARIANT=orca \
# python scripts/reinforcement_learning/rlinf/play.py \
#   --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
#   --config_name isaaclab_ppo_gr00t_assemble_trocar \
#   --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps_new_trocar_460samples \
#   --num_envs 128 \
#   --video \
#   2>&1 | tee play_128env_450steps_yiheng_rlinf_orca2.log

RAY_ADDRESS=local RAY_TMPDIR=/tmp/yh_orca1 \
ASSEMBLE_TROCAR_SCENE_VARIANT=orca \
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar_ac8 \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps_new_trocar_460samples \
  --num_envs 128 \
  --video \
  2>&1 | tee play_128env_450steps_yiheng_rlinf_orca2_ac8.log

RAY_ADDRESS=local RAY_TMPDIR=/tmp/yh_orca1 \
ASSEMBLE_TROCAR_SCENE_VARIANT=orca \
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar_ac8 \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps_new_trocar_460samples \
  --num_envs 128 \
  --video \
  2>&1 | tee play_128env_450steps_yiheng_rlinf_orca3_ac8.log

#----------------------------------------------------------

# RAY_ADDRESS=local RAY_TMPDIR=/tmp/yh_factory1 \
# ASSEMBLE_TROCAR_SCENE_VARIANT=factory \
# python scripts/reinforcement_learning/rlinf/play.py \
#   --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
#   --config_name isaaclab_ppo_gr00t_assemble_trocar \
#   --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps_new_trocar_460samples \
#   --num_envs 128 \
#   --video \
#   2>&1 | tee play_128env_450steps_yiheng_rlinf_factory2.log

RAY_ADDRESS=local RAY_TMPDIR=/tmp/yh_factory1 \
ASSEMBLE_TROCAR_SCENE_VARIANT=factory \
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar_ac8 \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps_new_trocar_460samples \
  --num_envs 128 \
  --video \
  2>&1 | tee play_128env_450steps_yiheng_rlinf_factory2_ac8.log

RAY_ADDRESS=local RAY_TMPDIR=/tmp/yh_factory1 \
ASSEMBLE_TROCAR_SCENE_VARIANT=factory \
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar_ac8 \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps_new_trocar_460samples \
  --num_envs 128 \
  --video \
  2>&1 | tee play_128env_450steps_yiheng_rlinf_factory3_ac8.log

RAY_ADDRESS=local RAY_TMPDIR=/tmp/yh_factory1 \
ASSEMBLE_TROCAR_SCENE_VARIANT=factory \
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar_ac8 \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps_new_trocar_460samples \
  --num_envs 128 \
  --video \
  2>&1 | tee play_128env_450steps_yiheng_rlinf_factory4_ac8.log

#----------------------------------------------------------

# RAY_ADDRESS=local RAY_TMPDIR=/tmp/yh_surgical1 \
# ASSEMBLE_TROCAR_SCENE_VARIANT=surgical \
# python scripts/reinforcement_learning/rlinf/play.py \
#   --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
#   --config_name isaaclab_ppo_gr00t_assemble_trocar \
#   --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps_new_trocar_460samples \
#   --num_envs 128 \
#   --video \
#   2>&1 | tee play_128env_450steps_yiheng_rlinf_surgical2.log

RAY_ADDRESS=local RAY_TMPDIR=/tmp/yh_surgical1 \
ASSEMBLE_TROCAR_SCENE_VARIANT=surgical \
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar_ac8 \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps_new_trocar_460samples \
  --num_envs 128 \
  --video \
  2>&1 | tee play_128env_450steps_yiheng_rlinf_surgical2_ac8.log

RAY_ADDRESS=local RAY_TMPDIR=/tmp/yh_surgical1 \
ASSEMBLE_TROCAR_SCENE_VARIANT=surgical \
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar_ac8 \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps_new_trocar_460samples \
  --num_envs 128 \
  --video \
  2>&1 | tee play_128env_450steps_yiheng_rlinf_surgical3_ac8.log

RAY_ADDRESS=local RAY_TMPDIR=/tmp/yh_surgical1 \
ASSEMBLE_TROCAR_SCENE_VARIANT=surgical \
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar_ac8 \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps_new_trocar_460samples \
  --num_envs 128 \
  --video \
  2>&1 | tee play_128env_450steps_yiheng_rlinf_surgical4_ac8.log

#----------------------------------------------------------