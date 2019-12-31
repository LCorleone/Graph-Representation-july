# -*- coding: utf-8 -*-
# @Author: lxy
# @Date:   2019-12-19 20:29:58
# @Last Modified by:   lxy
# @Last Modified time: 2019-12-30 16:46:23
from .unsupervised_node_classification import UnsupervisedNodeClassification
from .node_classification import NodeClassification


class Build_task(object):
    def __init__(self, args):
        self.args = args

    def build(self):
        if self.args.task == 'unsupervised_node_classification':
            return UnsupervisedNodeClassification(self.args)
        elif self.args.task == 'node_classification':
            return NodeClassification(self.args)