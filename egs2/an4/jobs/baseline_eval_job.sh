#!/usr/bin/env bash
#SBATCH --job-name=baseline_eval_job
#SBATCH --time=22:00:00
#SBATCH --nodes=1 --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --account=pas2400
#SBATCH --mail-type=ALL

module load gcc-compatibility/10.3.0
module load cmake/3.17.2
module load cuda/10.2.89
module load nccl/2.11.4
module load python
source activate espnet-env

set -x

source path.sh

/usr/bin/time ./run.sh --ngpu 1 --nj 16 \
              --inference_model latest.pth \
              --inference_args "--vocoder_tag parallel_wavegan/ljspeech_style_melgan.v1" \
              --inference_tag decode_with_ljspeech_style_melgan.v1 \
              --stage 7
