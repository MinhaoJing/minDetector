"""
    load last trained model
    :param model_path: folder path to search last model
    :return: iter, last model path(if none, return '')
"""

import glob as gb
import os.path as osp


def load_historical_model(model_path):
    """
    load last trained model
    """
    assert(osp.isdir(model_path))
    model_list = gb.glob(osp.join(model_path,'*.solverstate'))
    if model_list.__len__() == 0:
        return 0, ''

    iter_num = 0
    returned_model_path = ''
    for model_name in model_list:
        assert ('iter_' in model_name)
        tmp_iter = int(model_name.split('_')[-1].split('.')[0])
        if tmp_iter > iter_num:
            iter_num = tmp_iter
            returned_model_path = model_name

    assert (iter_num > 0) and (returned_model_path != '')
    return iter_num, returned_model_path