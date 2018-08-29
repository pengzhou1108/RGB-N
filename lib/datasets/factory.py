# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Peng Zhou
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
from datasets.casia import casia
from datasets.dist_fake import dist_fake
from datasets.nist import nist
from datasets.dvmm import dvmm
from datasets.swapme import swapme
import numpy as np

# Set up voc_<year>_<split> 
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    name = 'coco_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split> 
for year in ['2015']:
  for split in ['test', 'test-dev']:
    name = 'COCO_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: coco(split, year))

dvmm_path='/home-3/pengzhou@umd.edu/work/pengzhou/dataset/4cam_splc'
for split in ['train', 'test']:
    name = 'dist_{}'.format(split)
    __sets[name] = (lambda split=split: dvmm(split,2007,dvmm_path))

dso_path='/home-3/pengzhou@umd.edu/work/pengzhou/dataset/COVERAGE'
for split in ['cover_train_single', 'cover_test_single']:
    name = 'dist_{}'.format(split)
    __sets[name] = (lambda split=split: dist_fake(split,2007,dso_path))

nist_path='/home-3/pengzhou@umd.edu/work/pengzhou/dataset/NC2016_Test0613'
for split in ['NIST_train_new_2', 'NIST_test_new_2']:
    name = 'dist_{}'.format(split)
    __sets[name] = (lambda split=split: nist(split,2007,nist_path))

casia_path='/home-3/pengzhou@umd.edu/work/pengzhou/dataset/CASIA1'
for split in ['train_all_single', 'test_all_1']:
    name = 'casia_{}'.format(split)
    __sets[name] = (lambda split=split: casia(split,2007,casia_path))

coco_path='/home-3/pengzhou@umd.edu/work/pengzhou/dataset/cocostuff/coco/filter_tamper'
for split in ['train_filter', 'test_filter']:
    name = 'coco_{}'.format(split)
    __sets[name] = (lambda split=split: coco(split,2007,coco_path))

swapme_path='/home-3/pengzhou@umd.edu/work/xintong/medifor/images/dataset_1k_final'
for split in ['faceswap_rcnn_train_only', 'faceswap_rcnn_test']:
    name = 'face_{}'.format(split)
    __sets[name] = (lambda split=split: swapme(split,2007,swapme_path))


def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
