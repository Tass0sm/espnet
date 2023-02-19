#!/usr/bin/env bash
#SBATCH --job-name=eval_pretrained_model
#SBATCH --time=10:00:00
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

# ./run.sh --stage 1 --stop-stage 2
/usr/bin/time ./run.sh --ngpu 4 --stage 12  --use_streaming false --skip_data_prep true --skip_train true --download_model byan/librispeech_asr_train_asr_conformer_raw_bpe_batch_bins30000000_accum_grad3_optim_conflr0.001_sp
