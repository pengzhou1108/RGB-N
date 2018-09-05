#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
#from model.nms_wrapper import nms
from utils.cython_nms import nms
from utils.timer import Timer
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
from glob import glob
import re
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.resnet_v1_noise import resnet_noise
from nets.resnet_fusion import resnet_fusion
from scipy.io import savemat
import pdb

#CLASSES = ('__background__',
           #'aeroplane', 'bicycle', 'bird', 'boat',
           #'bottle', 'bus', 'car', 'cat', 'chair',
           #'cow', 'diningtable', 'dog', 'horse',
           #'motorbike', 'person', 'pottedplant',
           #'sheep', 'sofa', 'train', 'tvmonitor')
#CLASSES = ('__background__',
           #'tamper','authentic')

data_dir='/home-3/pengzhou@umd.edu/work/pengzhou/dataset/cocostuff/coco/filter_jpp_0_7'
data_dir_2='/home-3/pengzhou@umd.edu/work/pengzhou/dataset/cocostuff/coco/train2014'
#data_dir='/home-3/pengzhou@umd.edu/work/pengzhou/dataset/4cam_splc'
#data_dir_2='/home-3/pengzhou@umd.edu/work/pengzhou/dataset/4cam_auth'
data_dir='/home-3/pengzhou@umd.edu/work/pengzhou/dataset/NC2016_Test0613'
#data_dir='/home-3/pengzhou@umd.edu/work/pengzhou/dataset/dataset-dist'
#data_dir='/home-3/pengzhou@umd.edu/work/pengzhou/dataset/COVERAGE'
#data_dir='/home-3/pengzhou@umd.edu/work/xintong/medifor/portrait/test_data'
#data_dir='/home-3/pengzhou@umd.edu/work/pengzhou/dataset/CASIA1'
#data_dir='/home-3/pengzhou@umd.edu/work/xintong/medifor/portrait/u-net/model/rgbn/images'
vis_dir='output/filter_jpg_07'
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',),'coco_flip_0001_multi2_80k': ('res101_faster_rcnn_iter_96000.ckpt',),'coco_flip_00001_filter_80k': ('res101_faster_rcnn_iter_54000.ckpt',),\
'coco_flip_0001_n3_b64_80k': ('res101_noise_faster_rcnn_iter_84000.ckpt',),'coco_flip_0001_casia2_coco_au_80k': ('res101_faster_rcnn_iter_57000.ckpt',),'coco_flip_0001_casia2_coco_au_noise_80k': ('res101_noise_faster_rcnn_iter_80000.ckpt',),\
'coco_flip_00001_bilinear_casia2_finetune_80k': ('res101_fusion_faster_rcnn_iter_50000.ckpt',),'coco_flip_0001_noise_nist_finetune_80k':('res101_noise_faster_rcnn_iter_55000.ckpt',),'coco_flip_0001_filter_noise_2cls_80k':('res101_noise_faster_rcnn_iter_96000.ckpt',),\
'coco_flip_0001_bilinear_nist_new2_finetune': ('res101_fusion_faster_rcnn_iter_16000.ckpt',),'coco_flip_0001_bilinear_ssr_NIST_finetune_80k':('res101_fusion_faster_rcnn_iter_35000.ckpt',),'coco_flip_0001_nist_finetune_80k':('res101_faster_rcnn_iter_35000.ckpt',),\
'coco_flip_0001_bilinear_nist_3multi_80k': ('res101_fusion_faster_rcnn_iter_36000.ckpt',),'coco_flip_0001_cover_finetune_80k': ('res101_faster_rcnn_iter_10000.ckpt',),'coco_flip_0001_noise_cover_single_finetune':('res101_noise_faster_rcnn_iter_20000.ckpt',),'coco_flip_0001_bilinear_ssr_rpn_nonfix_80k': ('res101_fusion_faster_rcnn_iter_90000.ckpt',),\
'coco_flip_0001_bilinear_cover_single_finetune':('res101_fusion_faster_rcnn_iter_4000.ckpt',),'coco_flip_0001_bilinear_ssr_casia2_finetune_80k':('res101_fusion_faster_rcnn_iter_92000.ckpt',),'coco_flip_0001_casia2_finetune_all_80k':('res101_faster_rcnn_iter_50000.ckpt',),'coco_flip_0001_noise_casia2_single_finetune':('res101_noise_faster_rcnn_iter_110000.ckpt',),}
DATASETS= {'casia2_coco_au_noise': ('train_coco_au_2',),'casia2_coco_au': ('train_coco_au',),'train_all_2': ('train_all_2',),'train_all': ('train_all',),'train': ('train_multi_2',),'NIST_train_2': ('NIST_train_2',),'cover_train_single': ('cover_train_single',),'cover_train': ('cover_train',),'NIST_train': ('NIST_train',),'NIST_train_multi_single_new': ('NIST_train_multi_single_new',)\
,'fake_train': ('fake_train',),'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),'train_filter_2': ('train_filter_2',),'train_filter': ('train_filter',),'train_all_single': ('train_all_single',),'NIST_train_new_2':('NIST_train_new_2',),}
test_single=1
def vis_detections(im, class_name, dets, img_name,thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return 0

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    #fig=plt.figure()
    ax.imshow(im, aspect='equal')
    avg_score=0
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        avg_score=max(avg_score,score)
        #plt.imsave('{}.png'.format(os.path.join(vis_dir, img_name+'_'+str(score)+'_'+class_name)),im[int(max(0,bbox[0])):int(bbox[2]),max(0,int(bbox[1])):int(bbox[3]),:])
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
        #if class_name=='manipulation':
            #class_name='copy move'
        #ax.text(bbox[0], bbox[1] ,
                #'{:s} '.format(class_name),
                #bbox=dict(facecolor='blue', alpha=0.5),
                #fontsize=40, color='white')
    #avg_score=avg_score/len(inds)
    #ax.set_title(('{} detections with '
                 # 'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  #thresh),
                  #fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    #pdb.set_trace()
    plt.savefig('{}.png'.format(os.path.join(vis_dir, os.path.basename(img_name)+'_'+class_name+'_'+str(avg_score))))
    plt.close(fig)
    return avg_score
def debug(sess, net, image_name,bbox):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    if os.path.isfile(os.path.join(data_dir, image_name)):
        im_file = os.path.join(data_dir, image_name)

    else:
        im_file = os.path.join(data_dir_2, image_name)
    im = cv2.imread(im_file)
    im = im.astype(np.float32, copy=False)
    pixel_means=np.array([[[102.9801, 115.9465, 122.7717]]])
    revise=40
    #im[bbox[1]:bbox[1]+revise,bbox[0]:bbox[2],:]=pixel_means
    #im[bbox[1]:bbox[3],bbox[0]:bbox[0]+revise,:]=pixel_means
    #im[bbox[3]-revise:bbox[3],bbox[0]:bbox[2],:]=pixel_means
    #im[bbox[1]:bbox[3],bbox[2]-revise:bbox[2],:]=pixel_means
    im -= pixel_means

    blob=np.zeros((1,im.shape[0],im.shape[1],3),dtype=np.float)
    #pdb.set_trace()
    blob[0, 0:im.shape[0], 0:im.shape[1], :] = im
    #im = io.imread('http://mscoco.org/images/%d'%(int(image_name)))
    # Detect all object classes and regress object bounds
    try:
        c=np.zeros((11,5,5))
        c[0]=[[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]]
        c[0]=c[0]/12

        c[1][1][1]=-1
        c[1][1][2]=2
        c[1][1][3]=-1
        c[1][2][1]=2
        c[1][2][2]=-4
        c[1][2][3]=2
        c[1][3][1]=-1
        c[1][3][2]=2
        c[1][3][3]=-1
        c[1]=c[1]/4

        c[2][2][1]=1
        c[2][2][2]=-2
        c[2][2][3]=1
        c[2]=c[2]/2
        Wcnn=np.zeros((5,5,3,3))
        for i in xrange(3):
          #k=i%10+1
          #Wcnn[i]=[c[3*k-3],c[3*k-2],c[3*k-1]]
          Wcnn[:,:,0,i]=c[i]
          Wcnn[:,:,1,i]=c[i]
          Wcnn[:,:,2,i]=c[i]
        
        image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        conv = tf.nn.conv2d(image, Wcnn, [1, 1, 1, 1], padding='SAME',name='srm')
        head=sess.run(conv,feed_dict={image:blob})
        img=head[0,:,:,::-1]
        #pdb.set_trace()
        #head,conv = debug_filter(sess, net, im)
    except Exception as e:
        print(e)
        return
    return img
def cam(sess, net, image_name,bbox):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    if os.path.isfile(os.path.join(data_dir, image_name)):
        im_file = os.path.join(data_dir, image_name)

    else:
        im_file = os.path.join(data_dir_2, image_name)
    im = cv2.imread(im_file)
    im = im.astype(np.float32, copy=False)
    pixel_means=np.array([[[102.9801, 115.9465, 122.7717]]])
    revise=40
    #im[bbox[1]:bbox[1]+revise,bbox[0]:bbox[2],:]=pixel_means
    #im[bbox[1]:bbox[3],bbox[0]:bbox[0]+revise,:]=pixel_means
    #im[bbox[3]-revise:bbox[3],bbox[0]:bbox[2],:]=pixel_means
    #im[bbox[1]:bbox[3],bbox[2]-revise:bbox[2],:]=pixel_means
    #im -= pixel_means
    scores, boxes,feat,s1 = im_detect(sess, net, im)
    print(s1.shape)

    #rgb
    
    CONF_THRESH = 0.1
    NMS_THRESH = 0.2
    
    for cls_ind, cls in enumerate(CLASSES[1:]):
        if cls=='authentic':
            continue
        cls_ind += 1 # because we skipped background
        #cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        #cls_scores = scores[:, cls_ind]
        cam=np.dot(feat, s1[:,cls_ind])
        #pdb.set_trace()
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        #keep = nms(dets, NMS_THRESH,False)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        cam= cam[keep,:]

    inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
    if len(inds) == 0:
        return 0

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots()
    #fig=plt.figure()
    #ax.imshow(im, aspect='equal')
    avg_score=0
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        cam_i =cam[i,:,:]
        #pdb.set_trace()
        cam_i[np.where(cam_i>10)]=10
        cam_i[np.where(cam_i<0)]=0
        

        cam_i=(cam_i-np.min(cam_i))/np.max(cam_i)
        shape=np.shape(im[int(max(0,bbox[1])):int(bbox[3]),max(0,int(bbox[0])):int(bbox[2]),:])
        #heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
        heatmap_x = np.round(cv2.resize(cam_i,(shape[1],shape[0]))*255).astype(np.uint8)
        heatmap=cv2.applyColorMap(heatmap_x, cv2.COLORMAP_JET)
        #pdb.set_trace()
        avg_score=max(avg_score,score)
        #plt.imsave('{}.png'.format(os.path.join('.', image_name)),im[int(max(0,bbox[1])):int(bbox[3]),max(0,int(bbox[0])):int(bbox[2]),:].astype(np.uint8))
        im[int(max(0,bbox[1])):int(bbox[3]),max(0,int(bbox[0])):int(bbox[2]),:]=0.5*im[int(max(0,bbox[1])):int(bbox[3]),max(0,int(bbox[0])):int(bbox[2]),:]+0.5*heatmap
        
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format('tamper', score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
        #if class_name=='manipulation':
            #class_name='copy move'
        #ax.text(bbox[0], bbox[1] - 2,
                #'{:s} '.format(class_name),
                #bbox=dict(facecolor='blue', alpha=0.5),
                #fontsize=20, color='white')
    #avg_score=avg_score/len(inds)
    #ax.set_title(('{} detections with '
                 # 'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  #thresh),
                  #fontsize=14)
    ax.imshow(im.astype(np.uint8),aspect='equal')
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    #pdb.set_trace()
    plt.savefig('{}.png'.format(os.path.join(vis_dir, os.path.basename(image_name)+'_cam'+'_'+str(avg_score))))
    plt.close(fig)
        #im_score=vis_detections(im, cls, dets,image_name, thresh=CONF_THRESH)

        #pdb.set_trace()
        #head,conv = debug_filter(sess, net, im)
    #except Exception as e:
        #print(e)
        #return
    return im
def demo(sess, net, image_name,bbox):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    #im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    if os.path.isfile(os.path.join(data_dir, image_name)):
        im_file = os.path.join(data_dir, image_name)

    else:
        im_file = os.path.join(data_dir_2, image_name)
    revise=40
    im = cv2.imread(im_file) 
    #cv2.imwrite('b.jpg',im,[cv2.IMWRITE_JPEG_QUALITY, 70])
    #pdb.set_trace()
    #im=cv2.imread('b.jpg')
    pixel_means=np.array([[[102, 115, 122]]])
    #pdb.set_trace()
    #im[bbox[1]:bbox[1]+revise,bbox[0]:bbox[2],:]=pixel_means
    #im[bbox[1]:bbox[3],bbox[0]:bbox[0]+revise,:]=pixel_means
    #im[bbox[3]-revise:bbox[3],bbox[0]:bbox[2],:]=pixel_means
    #im[bbox[1]:bbox[3],bbox[2]-revise:bbox[2],:]=pixel_means
    #im[bbox[1]:bbox[3],bbox[0]:bbox[2],:]=pixel_means
    
    #im = io.imread('http://mscoco.org/images/%d'%(int(image_name)))
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    try:
        scores, boxes,_,_ = im_detect(sess, net, im)
    except Exception as e:
        print(e)
        return
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.01
    NMS_THRESH = 0.2
    
    for cls_ind, cls in enumerate(CLASSES[1:]):
        if cls=='authentic':
            continue
        cls_ind += 1 # because we skipped background
        #cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        #cls_scores = scores[:, cls_ind]
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        #keep = nms(dets, NMS_THRESH,False)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #im_score=vis_detections(cv2.resize(im,None,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR), cls, dets,image_name, thresh=CONF_THRESH)
        im_score=vis_detections(im, cls, dets,image_name, thresh=CONF_THRESH)
    return im_score  
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    #tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              #NETS[demonet][0])
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'EXP_DIR_'+demonet,
                              NETS[demonet][0])
    if 'noise' in NETS[demonet][0].split('_'):
        CLASSES = ('authentic',
           'tamper')
    else:
        CLASSES = ('__background__',
           'splicing','removal','manipulation')
        CLASSES = ('__background__',
            'tamper',
           'authentic')
        CLASSES = ('authentic',
           'tamper')           
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    elif demonet == 'res101':
        net = resnetv1(batch_size=1, num_layers=101)
    elif demonet == 'coco_flip_0001_new_80k':
        net = resnetv1(batch_size=1, num_layers=101)
    elif demonet == 'coco_flip_0001_multi2_80k':
        net = resnetv1(batch_size=1, num_layers=101)
    elif demonet == 'coco_flip_0001_casia2_coco_au_80k':
        net = resnetv1(batch_size=1, num_layers=101)
    elif demonet == 'coco_flip_0001_nist_finetune_80k':
        net = resnetv1(batch_size=1, num_layers=101)
    elif demonet == 'coco_flip_0001_cover_finetune_80k':
        net = resnetv1(batch_size=1, num_layers=101)
    elif demonet == 'coco_flip_0001_n3_b64_80k':
        net = resnet_noise(batch_size=1, num_layers=101)
    elif demonet == 'coco_flip_0001_casia2_coco_au_noise_80k':
        net = resnet_noise(batch_size=1, num_layers=101)
    elif demonet == 'coco_flip_00001_bilinear_casia2_finetune_80k':
        net = resnet_fusion(batch_size=1, num_layers=101)
    elif demonet == 'coco_flip_0001_noise_cover_single_finetune':
        net = resnet_noise(batch_size=1, num_layers=101)
    elif demonet == 'coco_flip_0001_noise_nist_finetune_80k':
        net = resnet_noise(batch_size=1, num_layers=101)
    elif demonet == 'coco_flip_0001_bilinear_nist_new2_finetune':
        net = resnet_fusion(batch_size=1, num_layers=101)
    elif demonet == 'coco_flip_0001_bilinear_ssr_NIST_finetune_80k':
        net = resnet_fusion(batch_size=1, num_layers=101)
    elif demonet == 'coco_flip_0001_bilinear_nist_3multi_80k':
        net = resnet_fusion(batch_size=1, num_layers=101)
    elif demonet == 'coco_flip_0001_bilinear_cover_single_finetune':
        net = resnet_fusion(batch_size=1, num_layers=101)
    elif demonet == 'coco_flip_0001_bilinear_ssr_casia2_finetune_80k':
        net = resnet_fusion(batch_size=1, num_layers=101)
    elif demonet == 'coco_flip_0001_bilinear_ssr_rpn_nonfix_80k':
        net = resnet_fusion(batch_size=1, num_layers=101)
    elif demonet == 'coco_flip_0001_casia2_finetune_all_80k':
        net = resnetv1(batch_size=1, num_layers=101) 
    elif demonet == 'coco_flip_0001_noise_casia2_single_finetune':
        net = resnet_noise(batch_size=1, num_layers=101)
    elif demonet == 'coco_flip_0001_filter_noise_2cls_80k':
        net = resnet_noise(batch_size=1, num_layers=101)   
    elif demonet == 'coco_flip_00001_filter_80k':
        net = resnetv1(batch_size=1, num_layers=101)              
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", len(CLASSES),
                          tag='default', anchor_scales=[8,16,32,64],
                          anchor_ratios=[0.5,1,2])
    saver = tf.train.Saver()
    #print(tfmodel)
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    #im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                #'001763.jpg', '004545.jpg']
    #for im_name in im_names:
        #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        #print('Demo for data/demo/{}'.format(im_name))
        #demo(sess, net, im_name)
    print('~~~~~~~~~~~~~~~start~~~~~~~~~~~~~~~~~~~~')
    score_save=[]
    label=[]
    save_name=[]
    if test_single:
        im_names=['probe/NC2016_9347.jpg']
        for im_name in im_names:
            if not os.path.isfile('{}.png'.format(os.path.join(vis_dir, im_name))):
                demo(sess, net, im_name,[])
                #cam(sess, net, im_name,[])
                #feat=debug(sess, net, im_name,[]) 
                #plt.imsave('{:s}.png'.format(im_name),feat.astype(np.uint8))       
    else:
        with open(os.path.join(data_dir, 'test_filter_2.txt'),'r') as f:
        #with open(os.path.join('/home-3/pengzhou@umd.edu/work/pengzhou/dataset', 'test.txt'),'r') as f:
            im_names=f.readlines()
            for file in im_names:
                im_name=file.split(' ')[0]
                im_box=[float(file.split(' ')[i]) for i in range(1,5)]
                im_label=int(file.strip().split(' ')[-1]=='tamper')
                #im_label=int(im_name.split('_')[0]=='Tp')
                print('Demo for data/demo/{}'.format(im_name))
                if not os.path.isfile('{}.png'.format(os.path.join(vis_dir, im_name))):
                    im_score=demo(sess, net, im_name,im_box)
                    #feat=debug(sess, net, im_name,im_box)
                    #score_save.append(im_score)
                    #cam(sess, net, im_name,[])
                    label.append(im_label)
                    save_name.append(im_name)
                    #plt.imsave(os.path.join(vis_dir,'{:s}'.format(im_name.split('/')[1])),feat.astype(np.uint8))
                    #break
        #savemat('auc_result_dvmm.mat',{'score':score_save,'label':label,'im_name':save_name})

    #filenames=glob(os.path.join(vis_dir, '*.png'))
    #for f in filenames:
        #img=os.path.splitext(os.path.basename(f))[0]
        #try:
            #im_name='Au'+'_'+''.join(re.sub(r'[^A-Za-z]+', '',img.split('_')[6]))+'_'+''.join(re.findall('\d+',img.split('_')[6]))+'.jpg'
            
            #im_name='COCO_train2014_{:012d}.jpg'.format(int(img.split('_')[1]))
     
            #print(im_name)
        #except:
            #print(img)
            #continue
        #if not os.path.isfile('{}.png'.format(os.path.join(vis_dir, im_name))):
            #demo(sess, net, im_name)
        
    print('~~~~~~~~~~~~~~~end~~~~~~~~~~~~~~~~~~~~')
    #plt.show()
