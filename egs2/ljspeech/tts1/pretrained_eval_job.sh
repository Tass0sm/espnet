#!/usr/bin/env bash
#SBATCH --job-name=ljspeech_whole_pretrained_job
#SBATCH --time=16:00:00
#SBATCH --nodes=1 --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=4
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

/usr/bin/time ./run.sh --download_model kan-bayashi/ljspeech_tacotron2 \
              --skip_train true
