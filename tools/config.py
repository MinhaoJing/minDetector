#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
    Initialization.
------------------------
    1.  Initialize params by:  from config import cfg
    2.  If need changing, write a config file (in yaml)
        and use cfg_from_file(yaml_file) to load it and
        override the default options.
    3.  some copied from fast_rcnn(by Ross Girshick)
"""

from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C

cfg.RNG_SEED = 3
cfg.GPU_ID = 0

# A small number that's used many times
__C.EPS = 1e-14

#
# Training options
#
__C.TRAIN = edict()

# paths
__C.TRAIN.ROOT_PATH = ""
__C.TRAIN.DATA_SET_NAME = "test_dataset"  # 当前的数据集的文件夹位置
__C.TRAIN.DATA_PATH = "./dataset"  # 存放不同数据集的文件夹位置
__C.TRAIN.DATA_SET_STATE = "train"  # train or test
# 图片后缀
__C.TRAIN.IMAGE_SUFFIX = ".jpg"
# 分类类别
__C.TRAIN.CLASSES = ('__background__',  # always index 0
                     'aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor')
# 获取Proposal的方法
__C.TRAIN.PROPOSAL_METHOD = 'gt'
# 是否筛取既没有正样本也没有负样本的图片
__C.TRAIN.FILTER_ROI = 'true'
# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5
# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.4
__C.TRAIN.BG_THRESH_LO = 0.1

# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = False
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)
# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH = 0.5

# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.
__C.TRAIN.ASPECT_GROUPING = True

__C.TRAIN.HAS_RPN = True
# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600,)

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000

__C.TRAIN.SOLVER_PATH = "./work_space/solvers"
__C.TRAIN.MODEL_DIR = "./work_space/models"
# pretrained model
__C.TRAIN.PRETRAINED_MODEL = ''

# image size
__C.TRAIN.WIDTH = 512
__C.TRAIN.HEIGHT = 512
# #images per batch
__C.TRAIN.IMS_PER_BATCH = 1
# batch size (#anchors)
__C.TRAIN.BATCH_SIZE = 128
# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 10000
# max iterations
__C.TRAIN.MAX_ITERS = 80000

##################################
#
# Testing options
#
##################################

__C.TEST = edict()

# image size
__C.TEST.WIDTH = 512
__C.TEST.HEIGHT = 512

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TEST.FG_THRESH = 0.5


# 根据input_cfg中的项对于cfg进行修改
def change_config(input_cfg):
    if hasattr(input_cfg, 'keys'):
        for change_items in input_cfg.keys():
            if hasattr(input_cfg[change_items], 'keys'):
                for _change_items in input_cfg[change_items].keys():
                    cfg[change_items][_change_items] = input_cfg[change_items][_change_items]
            else:
                cfg[change_items] = input_cfg[change_items]
    else:
        print " 错误！ 输入不是 easydict.EasyDict ! "
