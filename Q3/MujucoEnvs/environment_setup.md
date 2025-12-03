# Instructions


## You may use the following instructions to create a conda environment for stable baselines

1) Create a conda environment
• cd ./MujocoEnvs
• conda env create -f sb3 env lin.yml
• conda activate sb3
• pip install -e .
2) Install MujoCo
• wget https://mujoco.org/download/mujoco210-linux-x86 64.tar.gz
• tar -xzf mujoco210-linux-x86 64.tar.gz
3) Move the mujoco210 into /.mujoco/
• mkdir /.mujoco
• mv ./mujoco210 /.mujoco/