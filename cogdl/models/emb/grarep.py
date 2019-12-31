# -*- coding: utf-8 -*-
# @Author: lxy
# @Date:   2019-12-23 14:13:05
# @Last Modified by:   lxy
# @Last Modified time: 2019-12-23 14:14:55


import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import preprocessing


class GraRep(object):

    def __init__(self, args):
        super(GraRep, self).__init__()
        self.dimension = args.dimension
        self.step = args.step

    def train(self, G):
        self.G = G
        self.num_node = G.number_of_nodes()
        A = np.asarray(nx.adjacency_matrix(self.G).todense(), dtype=float)
        A = preprocessing.normalize(A, "l1")

        log_beta = np.log(1.0 / self.num_node)
        A_list = [A]
        T_list = [sum(A).tolist()]
        temp = A
        # calculate A^1, A^2, ... , A^step, respectively
        for i in range(self.step - 1):
            temp = temp.dot(A)
            A_list.append(A)
            T_list.append(sum(temp).tolist())

        final_emb = np.zeros((self.num_node, 1))
        for k in range(self.step):
            for j in range(A.shape[1]):
                A_list[k][:, j] = np.log(A_list[k][:, j] / T_list[k][j] + 1e-20) - log_beta
                for i in range(A.shape[0]):
                    A_list[k][i, j] = max(A_list[k][i, j], 0)
            # concatenate all k-step representations
            if k == 0:
                dimension = self.dimension - int(self.dimension / self.step) * (self.step - 1)
                final_emb = self._get_embedding(A_list[k], dimension)
            else:
                W = self._get_embedding(A_list[k], self.dimension / self.step)
                final_emb = np.hstack((final_emb, W))

        self.embeddings = final_emb
        return self.embeddings

    def _get_embedding(self, matrix, dimension):
        # get embedding from svd and process normalization for ut
        ut, s, _ = sp.linalg.svds(matrix, int(dimension))
        emb_matrix = ut * np.sqrt(s)
        emb_matrix = preprocessing.normalize(emb_matrix, "l2")
        return emb_matrix
