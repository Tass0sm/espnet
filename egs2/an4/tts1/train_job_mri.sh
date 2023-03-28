#!/bin/bash
#SBATCH --job-name=tts_task
#SBATCH --time=01:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH -p devel
#SBATCH --gpus-per-node=0
#SBATCH --mem=80gb


module load gcc-compatibility/10.3.0
module load cmake/3.17.2
module load cuda/10.2.89
module load nccl/2.11.4
module load python
module spider cuda
conda activate /home/karimimonsefi.1/espnet/tools/venv/envs/espnet

set -x

source path.sh

./run.sh --ngpu 1 --nj 16 --stage 6 --stop-stage 6 \
         --train_config "conf/train_transformer.yaml"
