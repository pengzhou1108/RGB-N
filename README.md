# RGB-N
Code and synthetic dataset generation for the CVPR 2018 paper "Learning Rich Features for Image Manipulation Detection" 

# Environment
tensorflow 0.12.1, python3.5.2, cuda 8.0.44 cudnn 5.1

other packages please run:
```
pip install -r requirements.txt
```

# Compile lib and compact_bilinear_pooling:
1. Check if the cuda lib path in `compact_bilinear_pooling/sequential_fft/complie.sh` is correct.

2. Run the command:
```
cd lib
make
cd compact_bilinear_pooling/sequential_fft
./compile.sh
```

For more detail, see https://github.com/ronghanghu/tensorflow_compact_bilinear_pooling


# Pre-trained model
For ImageNet resnet101 pre-trained model, please download from https://github.com/endernewton/tf-faster-rcnn

# Synthetic dataset 
1. download COCO 2014 dataset (http://cocodataset.org/#download) and COCO PythonAPI (https://github.com/cocodataset/cocoapi) and put in `coco_synthetic` folder. After this step the coco dataset folder 'cocostuff' will be created.
2. change `dataDir` in `coco_synthetic/demo.py` to the path of 'train2014' (e.g, `./cocostuff/coco/train2014`)
3. run `run_demo.sh 1 100` choose the begin and end COCO category used for creating the tamper synthetic dataset.
4. run `split_train_test.py` to make train/test split. (making sure that the images used to generate training set not overlap with the images for testing)

# Train on synthetic dataset
1. change the coco synthetic path in `lib/factory.py`:
```
coco_path= #FIXME
for split in ['coco_train_filter_single', 'coco_test_filter_single']:
    name = split
    __sets[name] = (lambda split=split: coco(split,2007,coco_path))
```
2. Specify the dataset, gpu, and network in `train_dist_faster.sh` as below as run the file
```
./train_faster_rcnn.sh 0 coco res101_fusion EXP_DIR coco_flip_0001_bilinear_new
```

# Citation:
If this code or dataset helps your research, please cite our paper:
```
@article{zhou2018learning,
  title={Learning Rich Features for Image Manipulation Detection},
  author={Zhou, Peng and Han, Xintong and Morariu, Vlad I and Davis, Larry S},
  journal={arXiv preprint arXiv:1805.04953},
  year={2018}
}
```