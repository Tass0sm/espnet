#!/bin/bash
#SBATCH --job-name=tts_task
#SBATCH --time=01:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH -p devel
#SBATCH --gpus-per-node=0
#SBATCH --mem=80gb

cd espnet
module spider cuda
conda activate conda activate speech_project_env

set -x

source path.sh

./run.sh --ngpu 1 --nj 16 --stage 6 --stop-stage 6 \
         --train_config "conf/train_transformer.yaml"
