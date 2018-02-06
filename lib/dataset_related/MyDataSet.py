#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from lib.dataset_related.imdb import imdb
from tools.config import cfg
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import cPickle


class MyDataSet(imdb):
    def __init__(self):
        imdb.__init__(self, cfg.TRAIN.DATA_SET_NAME)
        # train or test ?
        self._state = cfg.TRAIN.DATA_SET_STATE
        # 存放整个数据集的路径
        self._dataset_path = os.path.join(cfg.TRAIN.DATA_PATH, cfg.TRAIN.DATA_SET_NAME)
        # 存放整个数据集的路径整个数据集中DATA_SET_NAME数据的路径
        self._data_path = os.path.join(self._dataset_path, self._state)
        #
        self._classes = cfg.TRAIN.CLASSES
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

        self._image_ext = cfg.TRAIN.IMAGE_SUFFIX

        # self._data_path中所有图片不带后缀的名字
        self._image_name_list = self._load_image_name_list()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

        # PASCAL specific config options
        self.config = {#'cleanup': True,
                       #'use_salt': True,
                       'use_diff': False,
                       #'matlab_eval': False,
                       'rpn_file': None,
                       #'min_size': 2
                       }

        assert os.path.exists(self._dataset_path), \
            'dataset path does not exist: {}'.format(self._dataset_path)
        assert os.path.exists(self._data_path), \
            'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
            给定一个image_name_list中的标号，返回这张图片的路径
        """
        return self.image_path_from_name(self._image_name_list[i])

    def image_path_from_name(self, image_name):
        """
            给定一个图片不带后缀的名字，读取它的路径
        """
        image_path = os.path.join(self._data_path, image_name + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_name_list(self):
        """
            从self._data_path文件夹中读取所有后缀为self._image_ext的图片
            并将其名字(不带后缀)存入image_name_list
        """
        image_name_list = []
        suffix_len = len(self._image_ext)
        data_path = os.walk(self._data_path)
        for root, dirs, filelist in data_path:
            for filename in filelist:
                if filename.endswith(self._image_ext):
                    image_name_list.append(filename[:-suffix_len])
        return image_name_list

    def _get_default_path(self):
        """
            默认的数据集文件夹路径
        """
        work_dir = os.getcwd()
        return os.path.join(work_dir, 'dataset', self._name)

    def gt_roidb(self):
        """
            返回感兴趣区域数据
            This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_image_annotation(name)
                    for name in self._image_name_list]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def _load_image_annotation(self, name):
        """
            已经修改
            Load image and bounding boxes info from XML file in the PASCAL VOC
            format.
        """
        filename = os.path.join(self._dataset_path, 'Annotations', name + '.xml')
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
