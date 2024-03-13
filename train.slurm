#!/bin/bash
#SBATCH --nodes=1 # Get one node
#SBATCH --cpus-per-task=2 # Two cores per task
#SBATCH --ntasks=1 # But only one task
#SBATCH --gres=gpu:1 # And one GPU
#SBATCH --gres-flags=enforce-binding # Insist on good CPU/GPU alignment
#SBATCH --time=120:00:00 # Run for 10 minutes, at most
#SBATCH --job-name=bearmax_train # Name the job so I can see it in squeue
#SBATCH --error=bearmax_train.%J.err
#SBATCH --output=bearmax_train.%J.out

## Select python File to run
export file="train_and_val.py"

nvidia-smi && time python $file
sleep 3
