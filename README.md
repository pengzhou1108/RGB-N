# RGB-N

# Environment
tensorflow 0.12.1, python3.5.2, cuda 8.0.44 cudnn 5.1

# Pre-trained model
For ImageNet resnet101 pre-trained model, please download from https://github.com/endernewton/tf-faster-rcnn

# Synthetic dataset 
1. download COCO 2014 dataset (http://cocodataset.org/#download) and COCO PythonAPI (https://github.com/cocodataset/cocoapi) and put in `coco_synthetic` folder
2. change `dataDir` in `coco_synthetic/demo.py` to the path of 'train2014'
3. run `run_demo.sh`

# Train on synthetic dataset
1. change the coco synthetic path in `lib/factory.py`:
```
coco_path='/vulcan/scratch/pengzhou/dataset/filter_tamper' #FIXME
for split in ['train_filter_single', 'test_filter_single']:
    name = 'coco_{}'.format(split)
    __sets[name] = (lambda split=split: coco(split,2007,coco_path))
```
2. Specify the dataset, gpu, and network in `train_dist_faster.sh` as below as run the file
```
./train_faster_rcnn.sh 0 coco res101_fusion EXP_DIR coco_flip_0001_bilinear_new
```