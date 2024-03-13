#!/bin/bash
#SBATCH --nodes=1 # Get one node
#SBATCH --cpus-per-task=4 # Two cores per task
#SBATCH --mem-per-cpu=8192
#SBATCH --ntasks=1 # But only one task 
#SBATCH --gres=gpu:2 # And one GPU
#SBATCH --gres-flags=enforce-binding # Insist on good CPU/GPU alignment
#SBATCH --time=120:00:00 # Run for 10 minutes, at most
#SBATCH --job-name=bearmax_download # Name the job so I can see it in squeue
#SBATCH --error=bearmax_download.%J.err
#SBATCH --output=bearmax_download.%J.out

## Select python File to run
export file="download_data_google_open_images_v7_object_detection_dataset.py"

nvidia-smi && time python $file
sleep 3
