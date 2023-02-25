#!/usr/bin/env bash
#SBATCH --job-name=an4_train_baseline3_job
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --account=pas2400
#SBATCH --mail-type=ALL

module load gcc-compatibility/10.3.0
module load cmake/3.17.2
module load cuda/10.2.89
# module load nccl/2.11.4
module load python
source activate espnet-env

set -x

source path.sh

/usr/bin/time ./run.sh --stage 11 --stop-stage 12 \
              --asr_config conf/train_asr_transformer_variant2.yaml
