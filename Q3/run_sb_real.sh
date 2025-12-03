#MDIR="path to mujoco dir which contains mujoco210"

export MUJOCO_PY_MUJOCO_PATH=$MDIR/mujoco/mujoco210
export MUJOCO_PY_MJLIB_PATH=$MDIR/mujoco/mujoco210
export MPATH=$MDIR/mujoco/mujoco210/bin
export INCLUDE_PATHS="-I$MDIR/conda_env/include"
echo $INCLUDE_PATHS
export CPATH=$MDIR/conda_env/include:$CPATH
export C_INCLUDE_PATH=$MDIR/conda_env/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$MDIR/conda_env/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$MPATH:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/nvidia:$LD_LIBRARY_PATH
echo "LD_LIBRARY_PATH is now set to: $LD_LIBRARY_PATH"
# python MujucoEnvs/scripts/train_sb.py --env_name InvertedPendulum-v4
# python MujucoEnvs/scripts/train_sb.py --env_name Hopper-v4  
python MujucoEnvs/scripts/train_sb.py --env_name HalfCheetah-v4
