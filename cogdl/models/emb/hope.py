# -*- coding: utf-8 -*-
# @Author: lxy
# @Date:   2019-12-23 14:19:06
# @Last Modified by:   lxy
# @Last Modified time: 2019-12-23 14:21:55


import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import preprocessing


class HOPE(object):

    def __init__(self, args):
        super(HOPE, self).__init__()
        self.dimension = args.dimension
        self.beta = args.beta

    def train(self, G):
        self.G = G
        adj = nx.adjacency_matrix(self.G).todense()
        n = adj.shape[0]
        # The author claim that Katz has superior performance in related tasks
        # S_katz = (M_g)^-1 * M_l = (I - beta*A)^-1 * beta*A = (I - beta*A)^-1 * (I - (I -beta*A))
        #        = (I - beta*A)^-1 - I
        katz_matrix = np.asarray((np.eye(n) - self.beta * np.mat(adj)).I - np.eye(n))
        self.embeddings = self._get_embedding(katz_matrix, self.dimension)
        return self.embeddings

    def _get_embedding(self, matrix, dimension):
        # get embedding from svd and process normalization for ut and vt
        ut, s, vt = sp.linalg.svds(matrix, int(dimension / 2))
        emb_matrix_1, emb_matrix_2 = ut, vt.transpose()

        emb_matrix_1 = emb_matrix_1 * np.sqrt(s)
        emb_matrix_2 = emb_matrix_2 * np.sqrt(s)
        emb_matrix_1 = preprocessing.normalize(emb_matrix_1, "l2")
        emb_matrix_2 = preprocessing.normalize(emb_matrix_2, "l2")
        features = np.hstack((emb_matrix_1, emb_matrix_2))
        return features
