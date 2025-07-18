#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH -o /network/scratch/r/roger.creus-castanyer/slurm-%j.out # Write the log on scratch
#SBATCH --exclude=cn-j001


# 1. Load your environment
module load anaconda/3
conda activate /home/mila/r/roger.creus-castanyer/miniconda3/envs/trxl

wandb agent glen-berseth/ppo_trxl_adam/a263gjb2
