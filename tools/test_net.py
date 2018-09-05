# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi he, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test import test_net
from model.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.vgg16_noise import vgg16_noise
from nets.resnet_v1 import resnetv1
from nets.resnet_v1_noise import resnet_noise
from nets.resnet_v1_noise_init import resnet_noise_init
from nets.resnet_fusion import resnet_fusion
from nets.resnet_fusion_2rpn import resnet_fusion_2rpn
from nets.resnet_fusion_2rpn_sep import resnet_fusion_2rpn_sep
from nets.resnet_fusion_late_fusion import resnet_fusion_late_fusion
from nets.resnet_fusion_noise import resnet_fusion_noise
from nets.resnet_fusion_fix import resnet_fusion_fix
from nets.resnet_fusion_multi import resnet_fusion_multi
from tensorflow.python import pywrap_tensorflow
import pdb

def get_variables_in_checkpoint_file(file_name):
  try:
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    return var_to_shape_map 
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed "
            "with SNAPPY.")

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
            help='optional config file', default=None, type=str)
  parser.add_argument('--model', dest='model',
            help='model to test',
            default=None, type=str)
  parser.add_argument('--imdb', dest='imdb_name',
            help='dataset to test',
            default='voc_2007_test', type=str)
  parser.add_argument('--comp', dest='comp_mode', help='competition mode',
            action='store_true')
  parser.add_argument('--num_dets', dest='max_per_image',
            help='max number of detections per image',
            default=100, type=int)
  parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res50', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = parse_args()

  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  # if has model, get the name from it
  # if does not, then just use the inialization weights
  if args.model:
    filename = os.path.splitext(os.path.basename(args.model))[0]
  else:
    filename = os.path.splitext(os.path.basename(args.weight))[0]

  tag = args.tag
  tag = tag if tag else 'default'
  filename = tag + '/' + filename

  imdb = get_imdb(args.imdb_name)
  imdb.competition_mode(args.comp_mode)

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth=True

  # init session
  sess = tf.Session(config=tfconfig)
  # load network
  if args.net == 'vgg16':
    net = vgg16(batch_size=1)
  elif args.net == 'vgg16_noise':
    net = vgg16_noise(batch_size=1)
  elif args.net == 'res50':
    net = resnetv1(batch_size=1, num_layers=50)
  elif args.net == 'res50_noise':
    net = resnet_noise(batch_size=1, num_layers=50)
  elif args.net == 'res101':
    net = resnetv1(batch_size=1, num_layers=101)
  elif args.net == 'res101_noise':
    net = resnet_noise(batch_size=1, num_layers=101)
  elif args.net == 'res101_noise_init':
    net = resnet_noise_init(batch_size=1, num_layers=101)
  elif args.net == 'res101_fusion':
    net = resnet_fusion(batch_size=1, num_layers=101)
  elif args.net == 'res101_fusion_2rpn':
    net = resnet_fusion_2rpn(batch_size=1, num_layers=101)
  elif args.net == 'res101_fusion_2rpn_sep':
    net = resnet_fusion_2rpn_sep(batch_size=1, num_layers=101)
  elif args.net == 'res101_fusion_late_fusion':
    net = resnet_fusion_late_fusion(batch_size=1, num_layers=101)
  elif args.net == 'res101_fusion_noise':
    net = resnet_fusion_noise(batch_size=1, num_layers=101)
  elif args.net == 'res101_fusion_fix':
    net = resnet_fusion_fix(batch_size=1, num_layers=101)
  elif args.net == 'res101_fusion_multi':
    net = resnet_fusion_multi(batch_size=1, num_layers=101)
  elif args.net == 'res152':
    net = resnetv1(batch_size=1, num_layers=152)
  else:
    raise NotImplementedError

  # load model
  net.create_architecture(sess, "TEST", imdb.num_classes, tag='default',
                          anchor_scales=cfg.ANCHOR_SCALES,
                          anchor_ratios=cfg.ANCHOR_RATIOS)

  if args.model:
    print(('Loading model check point from {:s}').format(args.model))
    saver = tf.train.Saver()
    saver.restore(sess, args.model)
    #variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    #pdb.set_trace()
    #for v in variables:
      #print('Varibles: %s' % v.name)
    print('Loaded.')
  else:
    print(('Loading initial weights from {:s}').format(args.weight))
    sess.run(tf.global_variables_initializer())
    print('Loaded.')

  test_net(sess, net, imdb, filename, max_per_image=args.max_per_image)

  sess.close()
