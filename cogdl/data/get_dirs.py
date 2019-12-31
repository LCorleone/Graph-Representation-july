# -*- coding: utf-8 -*-
# @Author: lxy
# @Date:   2019-12-19 17:14:30
# @Last Modified by:   lxy
# @Last Modified time: 2019-12-19 17:17:12
import os.path as osp

def get_dir(path):
    return osp.expanduser(osp.normpath(path))