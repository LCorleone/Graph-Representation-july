import random
import numpy as np
import os.path as osp
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx

from para_config import Config
from cogdl.datasets import Build_dataset
from grave import plot_network, use_attributes
from tabulate import tabulate
from cogdl.data import get_dir
import pdb


def plot_graph(args):
    bd = Build_dataset(root=args.root_dataset)

    if not isinstance(args.dataset, list):
        args.dataset = [args.dataset]

    for name in args.dataset:
        nodes, edge_index, label = bd.get_data(name=name)

        depth = args.depth
        pic_file = get_dir(osp.join(args.save_dir, f'display_{name}.png'))
        if not osp.exists(get_dir(args.save_dir)):
            os.makedirs(get_dir(args.save_dir))

        col_names = ['Dataset', '#nodes', '#edges', '#features', '#classes']
        tab_data = [[name, nodes.shape[0], edge_index.shape[1], 0, label.shape[1]]]
        print(tabulate(tab_data, headers=col_names, tablefmt='psql'))

        G = nx.Graph()
        G.add_edges_from([tuple(edge_index[:, i]) for i in range(edge_index.shape[1])])

        s = random.choice(list(G.nodes()))
        q = [s]
        node_set = set([s])
        node_index = {s: 0}
        max_index = 1
        for _ in range(depth):
            nq = []
            for x in q:
                for key in G[x].keys():
                    if key not in node_set:
                        nq.append(key)
                        node_set.add(key)
                        node_index[key] = node_index[x] + 1
            if len(nq) > 0:
                max_index += 1
            q = nq

        cmap = cm.rainbow(np.linspace(0.0, 1.0, max_index))

        for node, index in node_index.items():
            G.nodes[node]['color'] = cmap[index]
            G.nodes[node]['size'] = (max_index - index) * 50

        fig, ax = plt.subplots()
        plot_network(G.subgraph(list(node_set)), node_style=use_attributes())
        plt.savefig(pic_file)
        plt.show()
        print(f'Sampled ego network saved to {pic_file} .')


if __name__ == '__main__':
    args = Config()
    args.set_model('fdf')
    args.set_dataset('wikipedia')
    args.depth = 1

    if isinstance(args.seed, list):
        args.seed = args.seed[0]

    random.seed(args.seed)
    np.random.seed(args.seed)

    plot_graph(args)
