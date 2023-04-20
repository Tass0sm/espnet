#!/usr/bin/env bash
#SBATCH --job-name=opencpop_download_and_prep
#SBATCH --time=8:00:00
#SBATCH --nodes=1 --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --account=pas2400
#SBATCH --mail-type=ALL

# module load gcc-compatibility/10.3.0
# module load cmake/3.17.2
# module load cuda/10.2.89
# module load nccl/2.11.4
# module load python
# source activate espnet-env

set -x

cd ../svs1
source path.sh

./run.sh --ngpu 0 --nj 16 --stop-stage 4
