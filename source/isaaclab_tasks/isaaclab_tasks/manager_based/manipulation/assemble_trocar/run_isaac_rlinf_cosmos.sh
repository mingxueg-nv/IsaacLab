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
# git clone https://github.com/isaac-sim/IsaacLab.git
# cd IsaacLab

# #### develop branch commit ####
# git switch develop
# git checkout e8d0f67689f
git clone https://github.com/mingxueg-nv/IsaacLab.git
git checkout pengfeig/cosmos_service

./isaaclab.sh -i

# ---- 3. Isaac-GR00T ----
git clone ssh://git@gitlab-master.nvidia.com:12051/dlmed/Isaac-GR00T.git
cd Isaac-GR00T
git checkout pg/dev
pip install -e .[base] --no-deps
# We are using Cosmos-transfer2.5 Release 1.5.1
bash scripts/install_cosmos_gr00t.sh
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

# ---- IsaacSim env vars (needed by AppLauncher) ----
export ISAAC_PATH=/localhome/local-pengfeig/pengfeig/isaac-sim-standalone-6.0.0-rc.22
export EXP_PATH=$ISAAC_PATH/apps
export CARB_APP_PATH=$ISAAC_PATH/kit

# ---- 6. start cosmos service (Transfer2.5) ----
export HF_HOME=/localhome/local-pengfeig/pengfeig/IsaacLab/Isaac-GR00T/third_party/cosmos-transfer2.5/cache
export HF_TOKEN="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
bash start_cosmos_workers.sh --num-gpus 1 # first one only for checkpoint download

# ---- 7. install pyzmq (for cosmos service) ----
pip install pyzmq

# ---- Run ----
# train
python scripts/reinforcement_learning/rlinf/train.py \
  --config_path /localhome/local-pengfeig/mingxue/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar \
  --model_path /localhome/local-pengfeig/mingxue/models/gr00t_sft/ \
  2>&1 | tee train.log

# play
python scripts/reinforcement_learning/rlinf/play.py \
  --config_path /localhome/local-pengfeig/mingxue/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/assemble_trocar/config \
  --config_name isaaclab_ppo_gr00t_assemble_trocar \
  --model_path /localhome/local-pengfeig/mingxue/models/rlinf \
  --num_envs 8 \
  --video \
  2>&1 | tee play.log
