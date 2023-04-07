#!/usr/bin/env bash
#SBATCH --job-name=an4_download_and_prep_job
#SBATCH --time=8:00:00
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

source path.sh

set -x

cd ../utts1

./run.sh --ngpu 0 --nj 16 --stage 0 --stop-stage 4
