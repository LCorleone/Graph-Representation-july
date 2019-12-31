# -*- coding: utf-8 -*-
# @Author: lxy
# @Date:   2019-12-19 11:43:24
# @Last Modified by:   lxy
# @Last Modified time: 2019-12-31 10:48:10


import numpy as np


class Config(object):
    def __init__(self):
        self.root_dataset = './dataset'
        # self.seed = [1]
        self.max_epoch = 10000
        self.patience = 500
        self.lr = 0.01
        self.weight_decay = 5e-4
        self.gpu = True
        self.dimension = 256
        self.hidden_size = 256
        self.num_shuffle = 5
        self.depth = 1  # depth of neighbours, for visualization
        self.device_id = '0'
        self.save_dir = './output'
        self.seed = np.random.randint(0, 10000, 1)
        print('ramdom seed is {}'.format(self.seed))

        # default
        self.model_list = ['deepwalk', 'grarep', 'hope', 'line', 'netmf', 'netsmf', 'node2vec', 'prone', 'gcn']

    def set_model(self, model):
        # 'node2vec'
        if model in self.model_list:
            self.model = model
            self.__set_model_para()
        else:
            raise Exception('invalid model name, please check if it is in {}'.format(self.model_list))

    def set_dataset(self, dataset):
        # 'wikipedia'
        self.dataset = dataset
        if self.dataset == 'cora':
            self.data_file = './dataset/raw/cora'
            self.label_size = 7

    def set_task(self, task):
        # 'node_classification'
        self.task = task

    def __set_model_para(self):
        if self.model == 'deepwalk':
            self.walk_length = 50  # Length of walk per source. Default is 80
            self.walk_num = 10  # Number of walks per source. Default is 40
            self.window_size = 10  # Window size of skip-gram model. Default is 5
            self.worker = 10  # Number of parallel workers. Default is 10
            self.iteration = 10  # Number of iterations. Default is 10
        elif self.model == 'node2vec':
            self.walk_length = 50  # Length of walk per source. Default is 80
            self.walk_num = 10  # Number of walks per source. Default is 40
            self.window_size = 10  # Window size of skip-gram model. Default is 5
            self.worker = 10  # Number of parallel workers. Default is 10
            self.iteration = 10  # Number of iterations. Default is 10
            self.p = 1.0  # Parameter in node2vec. Default is 1.0
            self.q = 1.0  # Parameter in node2vec. Default is 1.0
        elif self.model == 'netmf':
            self.window_size = 10
            self.rank = 256
            self.negative = 5
            self.is_large = True
        elif self.model == 'grarep':
            self.step = 5  # Number of matrix step in GraRep. Default is 5
        elif self.model == 'hope':
            self.beta = 0.01  # Parameter of katz for HOPE. Default is 0.01
        elif self.model == 'line':
            self.walk_length = 50  # Length of walk per source. Default is 50
            self.walk_num = 10  # Number of walks per source. Default is 20
            self.negative = 5  # Number of negative node in sampling. Default is 5
            self.batch_size = 100  # Batch size in SGD training process. Default is 1000
            self.alpha = 0.025  # Initial learning rate of SGD. Default is 0.025
            self.order = 3  # Order of proximity in LINE. Default is 3 for 1+2
        elif self.model == 'netsmf':
            self.window_size = 10  # Window size of approximate matrix. Default is 10
            self.negative = 5  # Number of negative node in sampling. Default is 1
            self.num_round = 100  # Number of round in NetSMF. Default is 100
            self.worker = 10  # Number of parallel workers. Default is 10
        elif self.model == 'prone':
            self.step = 5  # Number of items in the chebyshev expansion
            self.mu = 0.2
            self.theta = 0.5
        elif self.model == 'gcn':
            self.hidden_size = 16  # reset the hidden size for nn model

