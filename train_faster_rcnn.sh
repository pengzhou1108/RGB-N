#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  casia)
    TRAIN_IMDB="casia_train_all_single"
    TEST_IMDB="casia_test_all_single"
    STEPSIZE=40000
    ITERS=110000
    ANCHORS="[8,16,32,64]"
    RATIOS="[0.5,1,2]"
    ;;
  dist_fake)
    TRAIN_IMDB="dist_cover_train_single"
    TEST_IMDB="dist_cover_test_single"
    STEPSIZE=10000
    ITERS=25000
    ANCHORS="[8,16,32,64]"
    RATIOS="[0.5,1,2]"
    ;;
  NIST)
    TRAIN_IMDB="dist_NIST_train_new_2"
    TEST_IMDB="dist_NIST_test_new_2"
    STEPSIZE=30000
    ITERS=60000
    ANCHORS="[8,16,32,64]"
    RATIOS="[0.5,1,2]"
    ;;
  coco)
    TRAIN_IMDB="coco_train_filter_single"
    TEST_IMDB="coco_test_filter_single"
    STEPSIZE=40000
    ITERS=110000
    ANCHORS="[8,16,32,64]"
    RATIOS="[0.5,1,2]"
    ;;
  swapme)
    TRAIN_IMDB="face_faceswap_rcnn_train_only"
    TEST_IMDB="face_faceswap_rcnn_test"
    STEPSIZE=40000
    ITERS=60000
    ANCHORS="[8,16,32,64]"
    RATIOS="[0.5,1,2]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="./logs/${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
    NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.ckpt
else
    NET_FINAL=output/${NET}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_iter_${ITERS}.ckpt
fi
set -x

if [ ! -f ${NET_FINAL}.index ]; then
    if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
        python3 ./tools/trainval_net.py \
            --weight /vulcan/scratch/pengzhou/RGB-N/data/imagenet_weights/res101.ckpt \
            --imdb ${TRAIN_IMDB} \
            --imdbval ${TEST_IMDB} \
            --iters ${ITERS} \
            --cfg cfgs/${NET}.yml \
            --tag ${EXTRA_ARGS_SLUG} \
            --net ${NET} \
            --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
    else
        python3 ./tools/trainval_net.py \
            --weight /home-3/pengzhou@umd.edu/work/pengzhou/software/models/tf-faster-rcnn/data/imagenet_weights/${NET}.ckpt \
            --imdb ${TRAIN_IMDB} \
            --imdbval ${TEST_IMDB} \
            --iters ${ITERS} \
            --cfg cfgs/${NET}.yml \
            --net ${NET} \
            --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} TRAIN.STEPSIZE ${STEPSIZE} ${EXTRA_ARGS}
    fi
fi

##./test_faster_rcnn.sh $@
