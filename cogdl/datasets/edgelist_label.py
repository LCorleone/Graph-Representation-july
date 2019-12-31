import json
import os
import os.path as osp
import sys
from itertools import product

import networkx as nx
import numpy as np

from cogdl.data import download_url, get_dir
import pdb


def read_edgelist_label_data(folder, prefix, save_path):
    graph_path = get_dir(osp.join(folder, '{}.ungraph'.format(prefix)))
    cmty_path = get_dir(osp.join(folder, '{}.cmty'.format(prefix)))

    G = nx.read_edgelist(graph_path, nodetype=int, create_using=nx.Graph())
    num_node = G.number_of_nodes()
    nodes = np.array(list(G.nodes()))
    print('node number: ', num_node)
    with open(graph_path) as f:
        context = f.readlines()
        print('edge number: ', len(context))
        # tow line, each line has two corresponding node, total len(context) edges
        edge_index = np.zeros((2, len(context)))
        for i, line in enumerate(context):
            edge_index[:, i] = list(map(int, line.strip().split('\t')))

    with open(cmty_path) as f:
        context = f.readlines()
        print('class number: ', len(context))
        # if node m belongs to class k, thus (m, k) = 1
        label = np.zeros((num_node, len(context)))
        for i, line in enumerate(context):
            line = map(int, line.strip().split('\t'))
            for node in line:
                label[node, i] = 1
    np.savez(osp.join(save_path, prefix), nodes=nodes, edge_index=edge_index, label=label)


class Build_dataset(object):
    """docstring for build_dataset"""

    def __init__(self, root):
        super(Build_dataset, self).__init__()
        self.root = root
        self.raw_dir = osp.join(root, 'raw')
        self.processed = osp.join(root, 'processed')
        if not osp.exists(self.processed):
            os.makedirs(get_dir(self.processed))
        self.url = 'https://github.com/THUDM/ProNE/raw/master/data'

    def get_data(self, name):
        if not osp.exists(get_dir(osp.join(self.processed, name + '.npz'))):
            # raw_file_names = ['{}.{}'.format(s, f) for s, f in product([name], ['ungraph', 'cmty'])]
            # for file_name in raw_file_names:
            #     download_url('{}/{}'.format(self.url, file_name), self.raw_dir)
            read_edgelist_label_data(self.raw_dir, name, self.processed)

        npzfile = np.load(get_dir(osp.join(self.processed, name + '.npz')))
        return npzfile['nodes'], npzfile['edge_index'], npzfile['label']

