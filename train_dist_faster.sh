#!/bin/bash

#SBATCH -t 20:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu
#SBATCH --mem=20000
#SBATCH -p gpu
module load tensorflow
#source /lustre/pengzhou/software/caffe_mine/load_dependency.sh
./train_faster_rcnn.sh 0 NIST res101_fusion EXP_DIR coco_flip_0001_bilinear_new