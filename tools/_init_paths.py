"""
    Set up paths.
"""

import os.path
import sys
from tools.config import cfg

def _add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def init_paths(caffe_path=''):
    this_dir = os.path.dirname(__file__)
    if caffe_path != '':
        # Add caffe to PYTHONPATH
        caffe_path = os.path.join(caffe_path, 'python')
        _add_path(caffe_path)
    else:
        # Add caffe to PYTHONPATH
        caffe_path = os.path.join(this_dir, '..', 'caffe', 'python')
        _add_path(caffe_path)

    # # Add lib to PYTHONPATH
    # lib_path = osp.join(this_dir)
    # _add_path(lib_path)


def get_output_dir(net=None):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = os.path.abspath(os.path.join(cfg.TRAIN.ROOT_PATH, 'output', cfg.TRAIN.DATA_SET_NAME))
    if net is not None:
        outdir = os.path.join(outdir, net.name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir
