#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    prepare data for training and testing
"""

from lib.dataset_related.MyDataSet import MyDataSet
from lib.dataset_related.roidb import prepare_roidb, filter_roidb
from tools.config import cfg


def prepare_train_image():
    """
    input:  data path containing subfolders
            --sets, images, annotations
    return: image database (imdb) including
            image path list, corresponding gt
            and other info. needed
    """
    imdb = MyDataSet()

    # 目前只支持gt方法！！
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    imdb.gt_roidb()

    print 'Preparing training data...'
    prepare_roidb(imdb)
    print 'done'

    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    if cfg.TRAIN.FILTER_ROI and cfg.TRAIN.PROPOSAL_METHOD != 'gt':
        print 'Remove roidb entries that have no usable RoIs'
        filter_roidb(imdb.roidb)
        print 'done'

    return imdb.roidb, imdb
