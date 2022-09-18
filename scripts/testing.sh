module load python/3.10 StdEnv/2020 gcc/9.3.0 cuda/11.4 cudacore/.11.4.2 cudnn/8.2.0 scipy-stack/2020b glfw
#module load python/3.10 StdEnv/2020 cuda/11.4 cudacore/.11.4.2 cudnn/8.2.0 scipy-stack/2020b glfw
#module load gcc/11

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_PATH}/lib64:/cvmfs/soft.computecanada.ca/easybuild/software/2020/CUDA/cuda11.4/cudnn/8.2.0/lib64:/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/glfw/3.3.2/lib64

#export CUDA_DIR=${CUDA_PATH}
#module load gcc/11
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/home/jgsimard/.mujoco/mujoco210/bin:/usr/lib/nvidia

export MUJOCO_PATH=$HOME/.mujoco/mujoco210
export MUJOCO_PY_MUJOCO_PATH=$MUJOCO_PATH
export MJLIB_PATH=$MUJOCO_PATH/lib/libmujoco.so
export MUJOCO_GL="egl"

pushd $HOME
source jax/bin/activate
popd

#python -c "import mujoco_py"
#python train.py agent=sac env_name=HalfCheetah-v3 eval_interval=10000
python train.py agent=drq eval_interval=10000
