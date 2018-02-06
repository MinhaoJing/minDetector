#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    training
"""

import caffe
import os.path as osp
import google.protobuf as pb2
from caffe.proto import caffe_pb2
from tools.timer import Timer
from tools.config import cfg


from lib.load_historical_model import load_historical_model


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    """

    def __init__(self, roidb):
        """Initialize the SolverWrapper."""
        self.model_dir = cfg.TRAIN.MODEL_DIR
        self.pretrained_model = cfg.TRAIN.PRETRAINED_MODEL
        self.solver_path = cfg.TRAIN.SOLVER_PATH
        self.snapshot_iters = cfg.TRAIN.SNAPSHOT_ITERS
        self.max_iters = cfg.TRAIN.MAX_ITERS

        # solver
        assert (osp.exists(self.solver_path))
        self.solver = caffe.SGDSolver(self.solver_path)

        # historical model
        self.current_iter, self.historical_model = load_historical_model(self.model_dir)
        if self.current_iter > 0:
            print ('Loading historical model '
                   'weights from {:s}').format(self.historical_model)
            self.solver.restore(self.historical_model)

        'todo: solver.iter will be restored???'

        # pre-trained model
        if (self.current_iter == 0) and (self.pretrained_model != ''):
            assert osp.exists(self.pretrained_model)
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.solver.net.copy_from(self.pretrained_model)

        # solver params
        self.solver_param = caffe_pb2.SolverParameter()
        with open(self.solver_path, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        # 将roidb打乱顺序并输入到网络之中
        self.solver.net.layers[0].set_roidb(roidb)

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        self.solver.snapshot()
        print 'Wrote snapshot...'

    def train_model(self):
        """Network training loop."""
        timer = Timer()
        model_paths = []
        while self.solver.iter < self.max_iters:

            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % self.snapshot_iters == 0:
                model_paths.append(self.snapshot())


def train_net(roidb):
    """Train a network."""

    sw = SolverWrapper(roidb)

    print 'Solving...'
    sw.train_model()
    print 'done solving'

    # todo: plot loss curve & visualize result/val_out when training'
