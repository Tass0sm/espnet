#!/usr/bin/env bash
#SBATCH --job-name=download_librispeech
#SBATCH --time=05:00:00
#SBATCH --nodes=1 --ntasks-per-node=1
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

/usr/bin/time ./run.sh --stage 0 --stop-stage 2
