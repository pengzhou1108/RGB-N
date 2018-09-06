#!/bin/bash

#SBATCH -t 3:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu
#SBATCH --mem=30000
#SBATCH -p gpu
#SBATCH -g 1

module add cuda/8.0.44 cudnn/v5.1
module load Python3/3.5.2

./test_faster_rcnn.sh 0 coco res101_fusion EXP_DIR coco_flip_0001_bilinear_new
