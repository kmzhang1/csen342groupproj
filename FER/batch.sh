#!/bin/bash 
# 
#SBATCH --job-name=YourJobName
#SBATCH --output=somejob-%j.out 
# 
#SBATCH --partition=gpu
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4 
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00 
#
#SBATCH --mail-user=YourUserName@scu.edu
#SBATCH --mail-type=END
module load Anaconda3
conda activate YourCondaEnv
module purge

python train_fer.py --model MobileNetV2 --kd_T 5 -s 1 --mu 1.0
python train_fer.py --model MobileNetV2 --kd_T 5 -s 5 --mu 1.0
python train_fer.py --model MobileNetV2 --kd_T 5 -s 7 --mu 1.0