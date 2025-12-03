#!/bin/bash

# === SET THIS ===
MDIR="/home/aksh/.mujoco"

# ======== MUJOCO-PY PATHS (for mujoco 2.1.0) ========

export MUJOCO_PY_MUJOCO_PATH=$MDIR/mujoco210
export MUJOCO_PY_MJLIB_PATH=$MDIR/mujoco210
export MPATH=$MDIR/mujoco210/bin

export INCLUDE_PATHS="-I$HOME/miniconda3/envs/sb3/include"
echo $INCLUDE_PATHS

export CPATH=$HOME/miniconda3/envs/sb3/include:$CPATH
export C_INCLUDE_PATH=$HOME/miniconda3/envs/sb3/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$HOME/miniconda3/envs/sb3/include:$CPLUS_INCLUDE_PATH

# Add mujoco binaries to library path
export LD_LIBRARY_PATH=$MPATH:$LD_LIBRARY_PATH

# Add NVIDIA libs if needed
export LD_LIBRARY_PATH=/usr/lib/nvidia:$LD_LIBRARY_PATH

echo "MUJOCO_PY_MUJOCO_PATH = $MUJOCO_PY_MUJOCO_PATH"
echo "LD_LIBRARY_PATH is now set to: $LD_LIBRARY_PATH"
echo ""

# ============ TRAIN ENVIRONMENTS ============

#python3 MujucoEnvs/scripts/train_and_eval.py --env_name InvertedPendulum-v4 --algo PPO
python3 MujucoEnvs/scripts/train_and_eval.py --env_name InvertedPendulum-v4 --algo A2C

#python3 MujucoEnvs/scripts/train_and_eval.py --env_name Hopper-v4 --algo PPO
python3 MujucoEnvs/scripts/train_and_eval.py --env_name Hopper-v4 --algo A2C

python3 MujucoEnvs/scripts/train_and_eval.py --env_name HalfCheetah-v4 --algo PPO
python3 MujucoEnvs/scripts/train_and_eval.py --env_name HalfCheetah-v4 --algo A2C
