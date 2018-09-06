#!/bin/bash

current_node=$SLURM_NODEID
current_node=$(($current_node+1))
n_begin=$1
n_end=$2
#source /lustre/pengzhou/software/caffe_mine/load_dependency.sh
python demo.py --begin=$n_begin --end=$n_end
