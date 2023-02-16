#!/usr/bin/env bash
#SBATCH --job-name=train_model
#SBATCH --time=48:00:00
#SBATCH --nodes=1 --ntasks-per-node=32
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

# /usr/bin/time ./run.sh --ngpu 4 --skip_data_prep true --skip_train false
# stage 6 is calculating stats for the language model training text
# --stage 6 to include lm stats
/usr/bin/time ./run.sh --ngpu 4 --stage 7 --skip_eval true
