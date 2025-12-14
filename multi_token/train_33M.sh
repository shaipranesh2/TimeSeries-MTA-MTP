#!/bin/bash -l
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --time=23:59:00
#SBATCH --mem=300G
#SBATCH --partition=long
#SBATCH--job-name=33M-OpenLTM

# Echo time and hostname into log
echo "Date:     $(date)"
echo "Hostname: $(hostname)"
echo "33M-OpenLTM Training Started"
# This example uses Conda to manage package dependencies.
# See https://docs.mila.quebec/Userguide.html#conda for more information.
module load python/3.10
module load cuda/11.8
cd /home/mila/s/senthils/
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate /home/mila/s/senthils/miniconda3/envs/openLTM/
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="/network/scratch/s/senthils"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export WANDB_API_KEY=bdb5a3171cac37da8e7b7669f5c2b0e592d3122b
# Run the training script.
cd /network/scratch/s/senthils/multi_token/OpenLTM
export WANDB_RESUME=never
unset WANDB_RUN_ID
bash ./scripts/pretrain/timer_4.sh