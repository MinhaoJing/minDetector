
"""
    For training a new detector.
"""

# initialization
from config import cfg
# get data
from lib.train_related.prepare_train_data import get_data_with_gt
# set caffe path
import caffe

import numpy as np
from lib.train_related.train_net import train_net


def _init_caffe(cfg):
    """
        Initialize pycaffe in a training process.
    """
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)
    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)


if __name__ == '__main__':
    # get data and gt
    imdb = get_data_with_gt(cfg.TRAIN.DATA_PATH)

    # init caffe
    _init_caffe(cfg)

    #train net
    train_net(imdb, cfg)












