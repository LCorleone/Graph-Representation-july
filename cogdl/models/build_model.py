# -*- coding: utf-8 -*-
# @Author: lxy
# @Date:   2019-12-19 20:29:58
# @Last Modified by:   lxy
# @Last Modified time: 2019-12-30 16:00:59
from .emb.deepwalk import DeepWalk
from .emb.node2vec import Node2vec
from .emb.netmf import NetMF
from .emb.grarep import GraRep
from .emb.hope import HOPE
from .emb.line import LINE
from .emb.netsmf import NetSMF
from .emb.prone import ProNE
from .nn.gcn import GCN


class Build_model(object):
    def __init__(self, args):
        self.args = args

    def build(self):
        if self.args.model == 'deepwalk':
            return DeepWalk(self.args)
        elif self.args.model == 'node2vec':
            return Node2vec(self.args)
        elif self.args.model == 'netmf':
            return NetMF(self.args)
        elif self.args.model == 'grarep':
            return GraRep(self.args)
        elif self.args.model == 'hope':
            return HOPE(self.args)
        elif self.args.model == 'line':
            return LINE(self.args)
        elif self.args.model == 'netsmf':
            return NetSMF(self.args)
        elif self.args.model == 'prone':
            return ProNE(self.args)
        elif self.args.model == 'gcn':
            return GCN(self.args)
