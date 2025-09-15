#!/bin/bash

#SBATCH -p gpu
#SBATCH --gpus=a100:1
#SBATCH -t 2:00:00
#SBATCH --mem=16G
#SBATCH -J "train_darcy"
#SBATCH -o /dev/null
#SBATCH -e /dev/null

mkdir -p logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs/${TIMESTAMP}

exec > >(tee logs/${TIMESTAMP}/train_darcy_${SLURM_JOB_ID}.log)
exec 2> >(tee logs/${TIMESTAMP}/train_darcy_${SLURM_JOB_ID}.err >&2)

module load conda/latest
conda activate jaxnn
python darcy_train.py