# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Peng Zhou
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import pickle
import subprocess
import uuid
from .voc_eval import voc_eval
from model.config import cfg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class swapme(imdb):
  def __init__(self, image_set, year, dist_path=None):
    imdb.__init__(self, image_set)
    self._year = year
    self._image_set = image_set.split('face_')[1]
    self._dist_path = self._get_default_path() if dist_path is None \
                            else dist_path
    self._data_path=self._dist_path
    self._classes = ('__background__',  # always index 0
                     'tamper','authentic')
    #self._classes = ('authentic',  # always index 0
                     #'tamper')
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))
    self._image_ext = {'.png','.jpg','.tif','.bmp','.JPG'}
    self._image_index = self._load_image_set_index()
    # Default to roidb handler
    self._roidb_handler = self.gt_roidb

    assert os.path.exists(self._data_path), \
      'Path does not exist: {}'.format(self._data_path)

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """
    return self.image_path_from_index(os.path.splitext(self._image_index[i].split(' ')[0])[0])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    for ext in self._image_ext:
      #image_path = os.path.join('/home-3/pengzhou@umd.edu/work/xintong/medifor/portrait/test_data',
                              #index + ext)
      image_path = os.path.join(self._data_path,
                              index + ext)
      image_path1=os.path.join('/home-3/pengzhou@umd.edu/work/pengzhou/dataset/DATA2',
                              index.split('/')[1] + ext)
      if os.path.isfile(image_path):
        return image_path
      elif os.path.isfile(image_path1):
        return image_path1
      else:
        continue
    assert os.path.isfile(image_path) and os.path.isfile(image_path1), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def _load_image_set_index(self):
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
    image_set_file = os.path.join('/home-3/pengzhou@umd.edu/work/pengzhou/dataset',
                                  self._image_set + '.txt')
    assert os.path.exists(image_set_file), \
      'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
      image_index = [x.strip() for x in f.readlines()]
    #print(image_index)
    return image_index

  def _get_default_path(self):
    """
    Return the default path where PASCAL VOC is expected to be installed.
    """
    return os.path.join(cfg.DATA_DIR, 'swapme')

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.

    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if os.path.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        try:
          roidb = pickle.load(fid)
        except:
          roidb = pickle.load(fid, encoding='bytes')
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    gt_roidb = [self.roidb_gt(index)
                for index in self.image_index]
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))

    return gt_roidb

  def rpn_roidb(self):
    if int(self._year) == 2007 or self._image_set != 'test':
      gt_roidb = self.gt_roidb()
      rpn_roidb = self._load_rpn_roidb(gt_roidb)
      roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
    else:
      roidb = self._load_rpn_roidb(None)

    return roidb
  def roidb_gt(self,image_id):
    num_objs = int(len(image_id.split(' ')[1:])/5)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    # "Seg" area for pascal is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    # Load object bounding boxes into a data frame.
    for ix in range(num_objs):
      bbox = image_id.split(' ')[ix*5+1:ix*5+5]
      # Make pixel indexes 0-based
      x1 = float(bbox[0]) -1
      y1 = float(bbox[1]) -1 
      x2 = float(bbox[2]) -1 
      y2 = float(bbox[3]) -1
      if x1<0:
        x1=0
      if y1<0:
        y1=0 
      try:
        cls=self._class_to_ind[image_id.split(' ')[ix*5+5]]
      except:
        if int(image_id.split(' ')[ix*5+5])==0:
          print('authentic')
          cls=2
        else:
          cls = int(image_id.split(' ')[ix*5+5])
      boxes[ix, :] = [x1, y1, x2, y2]
      gt_classes[ix] = cls
      overlaps[ix, cls] = 1.0
      seg_areas[ix] = (x2 - x1 ) * (y2 - y1)
    #print(image_id)
    #print(boxes)
    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

  def _load_rpn_roidb(self, gt_roidb):
    filename = self.config['rpn_file']
    print('loading {}'.format(filename))
    assert os.path.exists(filename), \
      'rpn data not found at: {}'.format(filename)
    with open(filename, 'rb') as f:
      box_list = pickle.load(f)
    return self.create_roidb_from_box_list(box_list, gt_roidb)

  def _load_pascal_annotation(self, index):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
    tree = ET.parse(filename)
    objs = tree.findall('object')
    if not self.config['use_diff']:
      # Exclude the samples labeled as difficult
      non_diff_objs = [
        obj for obj in objs if int(obj.find('difficult').text) == 0]
      # if len(non_diff_objs) != len(objs):
      #     print 'Removed {} difficult objects'.format(
      #         len(objs) - len(non_diff_objs))
      objs = non_diff_objs
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    # "Seg" area for pascal is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
      bbox = obj.find('bndbox')
      # Make pixel indexes 0-based
      x1 = float(bbox.find('xmin').text) - 1
      y1 = float(bbox.find('ymin').text) - 1
      x2 = float(bbox.find('xmax').text) - 1
      y2 = float(bbox.find('ymax').text) - 1
      cls = self._class_to_ind[obj.find('name').text.lower().strip()]
      boxes[ix, :] = [x1, y1, x2, y2]
      gt_classes[ix] = cls
      overlaps[ix, cls] = 1.0
      seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

  def _get_comp_id(self):
    comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
               else self._comp_id)
    return comp_id

  def _get_voc_results_file_template(self):
    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    filename = 'casia_' + self._image_set + '_{:s}.txt'
    path = os.path.join(
      '.',
      filename)
    return path

  def _get_voc_noise_results_file_template(self):
    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    filename = 'casia_' + self._image_set + '_{:s}_noise.txt'
    path = os.path.join(
      '.',
      filename)
    return path

  def _write_voc_results_file(self, all_boxes):
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      print('Writing {} VOC results file'.format(cls))
      filename = self._get_voc_results_file_template().format(cls)
      print(filename)
      with open(filename, 'wt') as f:
        for im_ind, index in enumerate(self.image_index):
          dets = all_boxes[cls_ind][im_ind]
          if dets == []:
            continue
          # the VOCdevkit expects 1-based indices
          for k in range(dets.shape[0]):
            f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                    format(index.split(' ')[0], dets[k, -1],
                           dets[k, 0] + 1, dets[k, 1] + 1,
                           dets[k, 2] + 1, dets[k, 3] + 1))

  def _do_python_eval(self, output_dir='output'):
    annopath = os.path.join(
      '/home-3/pengzhou@umd.edu/work/pengzhou/dataset',
      'coco_multi' ,
      'Annotations',
      '{:s}.xml')
    imagesetfile = os.path.join(
      '/home-3/pengzhou@umd.edu/work/pengzhou/dataset',
      self._image_set + '.txt')
    cachedir = os.path.join('/home-3/pengzhou@umd.edu/work/pengzhou/dataset', 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    #use_07_metric = True if int(self._year) < 2010 else False
    use_07_metric = False
    print('dist metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
      os.mkdir(output_dir)
    for i, cls in enumerate(self._classes):
      if cls == '__background__' or cls == self.classes[0]:
        cls_ind=0
        continue
      else:
        cls_ind=self._class_to_ind[cls]
      #elif cls=='median_filtering':
        #cls_ind=3
        #continue
      filename = self._get_voc_results_file_template().format(cls)
      filename2 = self._get_voc_noise_results_file_template().format(cls)
      print(cls_ind)
      rec, prec, ap = voc_eval(
        filename,filename2, annopath, imagesetfile, cls_ind, cachedir, ovthresh=0.5,
        use_07_metric=use_07_metric,fuse=False)
      aps += [ap]
      print(('AP for {} = {:.4f},recall = {:.4f}, precision = {:.4f}'.format(cls, ap,rec[-1],prec[-1])))
      with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
        pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
      fig=plt.figure()
      plt.plot(rec,prec)
      fig.suptitle('PR curve for {} detection'.format(cls),fontsize=20)
      plt.xlabel('recall',fontsize=15)
      plt.xlim((0,1.0))
      plt.ylim((0,1.0))
      plt.ylabel('precision',fontsize=15)
      fig.savefig('{}.jpg'.format(cls))

    print(('Mean AP = {:.4f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
      print(('{:.3f}'.format(ap)))
    print(('{:.3f}'.format(np.mean(aps))))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')

  def _do_matlab_eval(self, output_dir='output'):
    print('-----------------------------------------------------')
    print('Computing results with the official MATLAB eval code.')
    print('-----------------------------------------------------')
    path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                        'VOCdevkit-matlab-wrapper')
    cmd = 'cd {} && '.format(path)
    cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
    cmd += '-r "dbstop if error; '
    cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
      .format(self._devkit_path, self._get_comp_id(),
              self._image_set, output_dir)
    print(('Running:\n{}'.format(cmd)))
    status = subprocess.call(cmd, shell=True)

  def evaluate_detections(self, all_boxes, output_dir):
    self._write_voc_results_file(all_boxes)
    self._do_python_eval(output_dir)
    #if self.config['matlab_eval']:
      #self._do_matlab_eval(output_dir)
    if self.config['cleanup']:
      for cls in self._classes:
        if cls == '__background__':
          continue
        filename = self._get_voc_results_file_template().format(cls)
        #os.remove(filename)

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True


if __name__ == '__main__':
  from datasets.swapme import swapme

  d = swapme('trainval', '2007')
  res = d.roidb
  from IPython import embed;

  embed()
