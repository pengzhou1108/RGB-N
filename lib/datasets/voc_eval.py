# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng Zhou
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
import pdb

def parse_rec(filename):
  """ Parse a PASCAL VOC xml file """
  tree = ET.parse(filename)
  objects = []
  for obj in tree.findall('object'):
    obj_struct = {}
    obj_struct['name'] = obj.find('name').text
    obj_struct['pose'] = obj.find('pose').text
    obj_struct['truncated'] = int(obj.find('truncated').text)
    obj_struct['difficult'] = int(obj.find('difficult').text)
    bbox = obj.find('bndbox')
    obj_struct['bbox'] = [int(bbox.find('xmin').text),
                          int(bbox.find('ymin').text),
                          int(bbox.find('xmax').text),
                          int(bbox.find('ymax').text)]
    objects.append(obj_struct)

  return objects


def voc_ap(rec, prec, use_07_metric=False):
  """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
  if use_07_metric:
    # 11 point metric
    ap = 0.
    for t in np.arange(0., 1.1, 0.1):
      if np.sum(rec >= t) == 0:
        p = 0
      else:
        p = np.max(prec[rec >= t])
      ap = ap + p / 11.
  else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
  return ap

def parse_txt(fileline,classname):
  #classes=('__background__','person_au','person_tp','airplane_tp','airplane_au','dog_tp','dog_au',
      #'train_tp','train_au','bed_tp','bed_au','refrigerator_tp','refrigerator_au')
  #classes=('__background__','person_au','person_tp','tv_tp','tv_au','airplane_tp','airplane_au','dog_tp','dog_au',
      #'bench_tp','bench_au','train_tp','train_au','broccoli_tp','broccoli_au','kite_tp','kite_au','bed_tp','bed_au','refrigerator_tp','refrigerator_au','bowl_tp','bowl_au')
  classes=('__background__', 'tamper','authentic')
  classes=('authentic', 'tamper')
  #classes=('__background__',  # always index 0
                     #'splicing','removal','manipulation')
  #classes=('__background__','person_au','tv_au','airplane_au','dog_au',
      #'bench_au','train_au','broccoli_au','kite_au','bed_au','refrigerator_au','bowl_au')
  class_to_ind = dict(list(zip(classes, list(range(len(classes))))))
  num_objs = int(len(fileline.split(' ')[1:])/5)
  objects=[]
  obj={}
  #print(fileline.split())
  #pdb.set_trace()
  #object['name']=fileline.split(" ")[0]
  for i in range(num_objs):
    obj['bbox']=[float(fileline.split(' ')[5*i+1]),
                  float(fileline.split(' ')[5*i+2]),
                  float(fileline.split(' ')[5*i+3]),
                  float(fileline.split(' ')[5*i+4])]
    try:
      obj['cls']=class_to_ind[fileline.split(' ')[5*i+5]]
    except:
      #pdb.set_trace()
      obj['cls']=int(fileline.split(' ')[5*i+5])
    #obj['bbox']=[int(fileline.split(' ')[(classname-1)*5+1]),
                    #int(fileline.split(' ')[(classname-1)*5+2]),
                    #int(fileline.split(' ')[(classname-1)*5+3]),
                    #int(fileline.split(' ')[(classname-1)*5+4])]
    #obj['cls']=int(fileline.split(' ')[(classname-1)*5+5])
    objects.append(obj.copy())
  return objects

def voc_eval(detpath,
            detpath2,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False,
             fuse=False):
  """rec, prec, ap = voc_eval(detpath,
                              annopath,
                              imagesetfile,
                              classname,
                              [ovthresh],
                              [use_07_metric])

  Top level function that does the PASCAL VOC evaluation.

  detpath: Path to detections
      detpath.format(classname) should produce the detection results file.
  annopath: Path to annotations
      annopath.format(imagename) should be the xml annotations file.
  imagesetfile: Text file containing the list of images, one image per line.
  classname: Category name (duh)
  cachedir: Directory for caching the annotations
  [ovthresh]: Overlap threshold (default = 0.5)
  [use_07_metric]: Whether to use VOC07's 11 point AP computation
      (default False)
  """
  # assumes detections are in detpath.format(classname)
  # assumes annotations are in annopath.format(imagename)
  # assumes imagesetfile is a text file with each line an image name
  # cachedir caches the annotations in a pickle file

  # first load gt
  if not os.path.isdir(cachedir):
    os.mkdir(cachedir)
  cachefile = os.path.join(cachedir, 'annots.pkl')

  # read list of images
  with open(imagesetfile, 'r') as f:
    lines = f.readlines()
  imagenames = [x.strip() for x in lines]

  if not os.path.isfile(cachefile):
    # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
      name=imagename.split(' ')[0]
      recs[name] = parse_txt(imagename,classname)
      #recs[imagename] = parse_rec(annopath.format(imagename))
      if i % 100 == 0:
        print('Reading annotation for {:d}/{:d}'.format(
          i + 1, len(imagenames)))
    # save
    print('Saving cached annotations to {:s}'.format(cachefile))
    #with open(cachefile, 'w') as f:
      #pickle.dump(recs, f)
  else:
    # load
    with open(cachefile, 'rb') as f:
      try:
        recs = pickle.load(f)
      except:
        recs = pickle.load(f, encoding='bytes')

  # extract gt objects for this class
  class_recs = {}
  npos = 0
  #pdb.set_trace()
  for imagename in imagenames:
    name=imagename.split(' ')[0]
    R = [obj for obj in recs[name] if obj['cls'] == classname]
    npos=npos+len(R)
    bbox = np.array([x['bbox'] for x in R])
    #difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
    det = [False] * len(R)
    class_recs[name] = {'bbox': bbox,
                             #'difficult': difficult,
                             'det': det}

  # read dets
  detfile = detpath.format(classname)
  detfile_n = detpath2.format(classname)
  #print(detfile)
  with open(detfile, 'r') as f:
    lines = f.readlines()
  if os.path.isfile(detfile_n):
    with open(detfile_n, 'r') as f_n:
      n_lines = f_n.readlines()
    n_splitlines = [x.strip().split(' ') for x in n_lines]
    #print(n_splitlines)
    image_n = [x[0] for x in n_splitlines]
    confidence_n = np.array([float(x[1]) for x in n_splitlines])
    BB_n = np.array([[float(z) for z in x[2:]] for x in n_splitlines])
  splitlines = [x.strip().split(' ') for x in lines]
  image_ids = [x[0] for x in splitlines]
  confidence = np.array([float(x[1]) for x in splitlines])
  BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
  count=np.zeros(10)
  noise_ct=0
  select_final=np.array([True]*len(image_ids))
  image_select=[]
  if BB.shape[0] > 0 and fuse:
    for k in range(len(image_ids)):
      if image_ids[k] in image_select:
        select_final[k]=False
        continue
      if image_ids[k] in image_n:
        bb = BB[k, :].astype(float)
        index=[i for i,ex in enumerate(image_n) if ex==image_ids[k]]
        bb1 = BB_n[index, :].astype(float)
        #print(index,bb1)
        #pdb.set_trace()
        c_n=confidence_n[index]
        

        ix_min = np.maximum(bb1[:, 0], bb[0])
        iy_min = np.maximum(bb1[:, 1], bb[1])
        ix_max = np.minimum(bb1[:, 2], bb[2])
        iy_max = np.minimum(bb1[:, 3], bb[3])
        iw = np.maximum(ix_max - ix_min + 1., 0.)
        ih = np.maximum(iy_max - iy_min + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (bb1[:, 2] - bb1[:, 0] + 1.) *
               (bb1[:, 3] - bb1[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ov_max = np.max(overlaps)
        jmax = np.argmax(overlaps)
        if ov_max>=0.5:
          count[int(ov_max*10)]=count[int(ov_max*10)]+1
          #print(confidence[k],c_n)
          #confidence[k]=np.maximum(confidence[k],c_n[jmax])
          confidence[k]=(confidence[k]+c_n[jmax])/2
          #pdb.set_trace()
          BB[k,:]=(confidence[k]*BB[k,:]+c_n[jmax]*bb1[jmax,:])/np.maximum(confidence[k]+c_n[jmax], np.finfo(np.float64).eps)
          image_select.append(image_ids[k])
          #if confidence[k]<c_n[jmax]-0.5:
            #BB[k,:]=bb1[jmax,:]
          #print(image_ids[k],confidence[k],c_n[jmax])
        elif ov_max<0.5 and ov_max>0.1:
          count[int(ov_max*10)]=count[int(ov_max*10)]+1
          image_select.append(image_ids[k])
          #select_final[k]=False
          #BB[k,:]=(confidence[k]*BB[k,:]+c_n[jmax]*bb1[jmax,:])/(confidence[k]+c_n[jmax])
          #pass
          #select_final[k]=False
          #confidence[k]=0.7*confidence[k]+0.3*c_n[jmax]
          #if confidence[k]<c_n[jmax]:
            #BB[k,:]=bb1[jmax,:]
          #print(image_ids[k],confidence[k],c_n[jmax])
          #confidence[k]=confidence[k]*max(ov_max+0.2,0.6)
        else:
          count[int(ov_max*10)]=count[int(ov_max*10)]+1
          select_final[k]=False
          #confidence[k]=confidence[k]*0.9
          #if confidence[k]<c_n[jmax]:
            #BB[k,:]=bb1[jmax,:]
            #confidence[k]=c_n[jmax]*0.9
    for nk in range(len(image_n)):
      if image_n[nk] not in image_ids:
        noise_ct=noise_ct+1
        #image_ids.append(image_n[nk])
        #select_final.append(select_final,True)
        #confidence=np.append(confidence,confidence_n[nk])
        #BB=np.vstack((BB,BB_n[nk,:]))
    print('rgb no overlap: {:s}'.format(count))
    print('noise no overlap: {:d}'.format(noise_ct))
  image_ids=np.extract(select_final,image_ids)
  confidence=np.extract(select_final,confidence)
  BB=BB[select_final,:]
  nd = len(image_ids)
  #print(image_ids)
  tp = np.zeros(nd)
  fp = np.zeros(nd)
  bb=[]
  bb1=[]
  #pdb.set_trace()
  if BB.shape[0] > 0:
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    #print(class_recs)
    #print(sorted_ind)
    # go down dets and mark TPs and FPs
    for d in range(nd):

      


      #print(bb1)
      R = class_recs[image_ids[d]]
      bb = BB[d, :].astype(float)
      if fuse:
        if image_ids[d] in image_n:
          index=[i for i,ex in enumerate(image_n) if ex==image_ids[d]]
          bb1 = BB_n[index, :].astype(float)
        ix_min = np.maximum(bb1[:, 0], bb[0])
        iy_min = np.maximum(bb1[:, 1], bb[1])
        ix_max = np.minimum(bb1[:, 2], bb[2])
        iy_max = np.minimum(bb1[:, 3], bb[3])
        iw_n = np.maximum(ix_max - ix_min + 1., 0.)
        ih_n = np.maximum(iy_max - iy_min + 1., 0.)
        inters_n = iw_n * ih_n

        # union
        un = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (bb1[:, 2] - bb1[:, 0] + 1.) *
               (bb1[:, 3] - bb1[:, 1] + 1.) - inters_n)

        overlaps_n = inters_n / un
        ov_max_n = np.max(overlaps_n)


      ovmax = -np.inf
      BBGT = R['bbox'].astype(float)
      #print(BBGT)
      #pdb.set_trace()
      if BBGT.size > 0:
        # compute overlaps
        # intersection
        #print(BBGT)
        #print(bb)
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)
        print("overlap:")
        print(overlaps)

      if ovmax > ovthresh:
        #print(R['det'][jmax])
        #if not R['difficult'][jmax]:
        if not R['det'][jmax]:
          #print(R['det'][jmax])
          tp[d] = 1.
          R['det'][jmax] = 1
        else:
          fp[d] = 1.
      else:
        print('fp:{:s}'.format(image_ids[d]))
        if fuse:
          print('score:{:f}, ovmax:{:f}'.format(-sorted_scores[d],ov_max_n))

        fp[d] = 1.

  # compute precision recall
  fp = np.cumsum(fp)
  tp = np.cumsum(tp)
  rec = tp / float(npos)
  # avoid divide by zero in case the first detection matches a difficult
  # ground truth
  prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
  ap = voc_ap(rec, prec, use_07_metric)

  return rec, prec, ap