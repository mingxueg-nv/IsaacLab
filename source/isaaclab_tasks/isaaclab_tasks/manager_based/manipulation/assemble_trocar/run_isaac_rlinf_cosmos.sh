# ============================================================
# IsaacLab + RLinf setup script
# ============================================================
ENV_NAME=isaaclab_develop_6.0_pg
conda create -y -n isaaclab_develop_6.0_pg python=3.12    # for isaacsim 6.0
conda activate isaaclab_develop_6.0_pg

python -m pip install --upgrade pip setuptools wheel

# ---- 1. IsaacSim standalone (must be set up BEFORE IsaacLab) ----
# isaacsim 6.0.0-rc.22 (standalone, python 3.12)
wget "https://d4i3qtqj3r0z5.cloudfront.net/isaac-sim-standalone%406.0.0-rc.22%2Brelease.33481.407f3ea1.gl.manylinux_2_35_x86_64.release.zip" \
     -O isaac-sim-standalone-6.0.0-rc.22.zip
unzip isaac-sim-standalone-6.0.0-rc.22.zip -d isaac-sim-standalone-6.0.0-rc.22
# Create .pth file
# Wires Isaac Sim standalone into the conda env so Python can find
sudo tee "/localhome/local-pengfeig/miniconda3/envs/isaaclab_develop_6.0_pg/lib/python3.12/site-packages/isaacsim-standalone.pth" > /dev/null << 'PTHEOF'
import os, ctypes, sys; _ir="/localhome/local-pengfeig/pengfeig/isaac-sim-standalone-6.0.0-rc.22"; _cl=os.path.join(sys.prefix,"lib"); _sp=os.path.join(sys.prefix,"lib/python3.12/site-packages"); os.environ.setdefault("ISAAC_PATH",_ir); os.environ.setdefault("EXP_PATH",os.path.join(_ir,"apps")); os.environ.setdefault("CARB_APP_PATH",os.path.join(_ir,"kit")); _nv=[os.path.join(_sp,"nvidia",d,"lib") for d in os.listdir(os.path.join(_sp,"nvidia")) if os.path.isdir(os.path.join(_sp,"nvidia",d,"lib"))] if os.path.isdir(os.path.join(_sp,"nvidia")) else []; _lp=os.pathsep.join([_cl,_ir,os.path.join(_ir,"kit"),os.path.join(_ir,"kit/kernel/plugins"),os.path.join(_ir,"kit/libs/iray"),os.path.join(_ir,"kit/plugins"),os.path.join(_ir,"kit/plugins/carb_gfx"),os.path.join(_ir,"kit/plugins/rtx"),os.path.join(_ir,"kit/plugins/gpu.foundation")]+_nv); os.environ["LD_LIBRARY_PATH"]=_lp+os.pathsep+os.environ.get("LD_LIBRARY_PATH",""); ctypes.CDLL(os.path.join(_cl,"libpython3.12.so.1.0")); ctypes.CDLL(os.path.join(_ir,"kit/libcarb.so"))
/localhome/local-pengfeig/pengfeig/isaac-sim-standalone-6.0.0-rc.22/python_packages
/localhome/local-pengfeig/pengfeig/isaac-sim-standalone-6.0.0-rc.22/exts/isaacsim.simulation_app
/localhome/local-pengfeig/pengfeig/isaac-sim-standalone-6.0.0-rc.22/kit/kernel/py
/localhome/local-pengfeig/pengfeig/isaac-sim-standalone-6.0.0-rc.22/kit/plugins/bindings-python
PTHEOF

# ---- 2. IsaacLab ----
git clone https://github.com/mingxueg-nv/IsaacLab.git
git checkout pengfeig/cosmos_service

./isaaclab.sh -i

# ---- 3. Isaac-GR00T ----
git clone ssh://git@gitlab-master.nvidia.com:12051/dlmed/Isaac-GR00T.git
cd Isaac-GR00T
git checkout pg/dev
pip install -e .[base] --no-deps
# We are using Cosmos-transfer2.5 Release 1.5.1
bash scripts/setup_z_image_venv.sh
cd ../

#

# ---- 4. RLinf ----
pip install -e "source/isaaclab_contrib[rlinf]" --ignore-requires-python  

# ---- 5. flash-attn (LAST, after all deps are finalized) ----
#pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"
MAX_JOBS=128 pip install flash-attn==2.8.3 --no-build-isolation --force-reinstall --no-deps

# ---- 6. install pyzmq (for cosmos service) ----
pip install pyzmq

# ---- IsaacSim env vars (needed by AppLauncher) ----
export ISAAC_PATH=/localhome/local-pengfeig/pengfeig/isaac-sim-standalone-6.0.0-rc.22
export EXP_PATH=$ISAAC_PATH/apps
export CARB_APP_PATH=$ISAAC_PATH/kit

# ---- 7. start cosmos service (Transfer2.5) ----
./start_z_image_workers.sh --num-gpus 8 --base-port 5657 --wait-ready --warmup



# ---- Run ----
# train
python scripts/reinforcement_learning/rlinf/train.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/orca-dev-test/rlinf/actor/model_state_dict \
  2>&1 | tee train_from_mingxue_0.6_rlinf_no_cosmos.log

python scripts/reinforcement_learning/rlinf/train.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar_z_image \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/orca-dev-test/rlinf/actor/model_state_dict \
  2>&1 | tee train_pg_64env_480_640_from_mingxue_0.6_rlinf_z_image.log

python scripts/reinforcement_learning/rlinf/train.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar_z_image \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/g1_install_trocar_sim_box_v3_60_train_bs32_1_gpus_cos_30k_tune_visual \
  2>&1 | tee train_pg_64env_480_640_z_image_from_yun_baseline.log

# stage 1 resume
python scripts/reinforcement_learning/rlinf/train.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar_z_image \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/g1_install_trocar_sim_box_v3_60_train_bs32_1_gpus_cos_30k_tune_visual \
  --resume_dir /localhome/local-pengfeig/pengfeig/IsaacLab/scripts/reinforcement_learning/rlinf/logs/rlinf/20260501-15:10:53-Isaac-Assemble-Trocar-G129-Dex3-RLinf-MultiModal-v0/test_gr00t/checkpoints/global_step_16 \
  2>&1 | tee train_pg_64env_480_640_z_image_from_yun_baseline_resume_gs16.log

# stage 2 resume
python scripts/reinforcement_learning/rlinf/train.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar_z_image \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/g1_install_trocar_sim_box_v3_60_train_bs32_1_gpus_cos_30k_tune_visual \
  --resume_dir /localhome/local-pengfeig/pengfeig/IsaacLab/scripts/reinforcement_learning/rlinf/logs/rlinf/20260502-20:54:43-Isaac-Assemble-Trocar-G129-Dex3-RLinf-MultiModal-v0/test_gr00t/checkpoints/global_step_40 \
  2>&1 | tee train_pg_stage2_64env_480_640_z_image_from_yun_baseline_resume_gs40.log

# stage 3 resume
python scripts/reinforcement_learning/rlinf/train.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar_z_image \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/g1_install_trocar_sim_box_v3_60_train_bs32_1_gpus_cos_30k_tune_visual \
  --resume_dir /localhome/local-pengfeig/pengfeig/IsaacLab/scripts/reinforcement_learning/rlinf/logs/rlinf/20260503-16:18:10-Isaac-Assemble-Trocar-G129-Dex3-RLinf-MultiModal-v0/test_gr00t/checkpoints/global_step_48 \
  2>&1 | tee train_pg_stage3_64env_480_640_z_image_from_yun_baseline_resume_gs48.log

# stage 4 resume
python scripts/reinforcement_learning/rlinf/train.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar_z_image \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/g1_install_trocar_sim_box_v3_60_train_bs32_1_gpus_cos_30k_tune_visual \
  --resume_dir /localhome/local-pengfeig/pengfeig/IsaacLab/scripts/reinforcement_learning/rlinf/logs/rlinf/20260503-19:35:44-Isaac-Assemble-Trocar-G129-Dex3-RLinf-MultiModal-v0/test_gr00t/checkpoints/global_step_56 \
  2>&1 | tee train_pg_stage4_64env_480_640_z_image_from_yun_baseline_resume_gs56.log

# play
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar_eval \
  --num_envs 64 \
  --video \
  2>&1 | tee play_64env_global_step_108_cosmos_augmented.log

python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps \
  --num_envs 8 \
  --video \
  2>&1 | tee play_8env_sim6_gr00t_n15_50ksteps.log

python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar_yiheng \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/sim6_gr00t_n15_50ksteps \
  --num_envs 8 \
  --video \
  2>&1 | tee play_8env_chunk4_sim6_gr00t_n15_50ksteps.log

python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/orca-dev-test/rlinf/actor/model_state_dict \
  --num_envs 16 \
  --video \
  2>&1 | tee play_16env_mingxue_0.6_rlinf.log

python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/pengfeig/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar \
  --model_path /localhome/local-pengfeig/pengfeig/models/gr00t/g1_install_trocar_sim_box_v3_60_train_bs32_1_gpus_cos_30k_tune_visual \
  --num_envs 64 \
  --video \
  2>&1 | tee play_64env_yun_baseline.log



