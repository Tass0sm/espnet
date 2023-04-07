#!/usr/bin/env bash
#SBATCH --job-name=an4_train_job
#SBATCH --time=16:00:00
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

EXP=$(realpath ../tts-project-exp1)
CONF=$(realpath ../utts1/conf/train_transformer.yaml)

cd ../tts1

./run.sh --ngpu 1 --nj 16 --stage 7 \
         --token_type char \
         --cleaner none \
         --g2p none \
         --expdir $EXP \
         --train_config $CONF \
         --inference_model "latest.pth"
