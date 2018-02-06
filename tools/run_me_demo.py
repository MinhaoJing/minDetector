#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tools import _init_paths
from lib.train_related.prepare_train_data import prepare_train_image
from lib.train_related.train_net import train_net
from tools.config import cfg, change_config
import argparse
import sys


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Faster R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--net_name', dest='net_name',
                        help='network name (e.g., "ZF")',
                        default=None, type=str)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # 这里应该输入caffe的路径，不输入就默认在当前文件夹下
    _init_paths.init_paths()

    #
    #change_config(cfg)

    # 构建output文件夹: 构建output文件夹: TRAIN.ROOT_PATH/ouput/TRAIN.DATA_SET_NAME
    _init_paths.get_output_dir()

    roidb, imdb = prepare_train_image()
    print "preprocessing is over!"

    train_net(roidb)
