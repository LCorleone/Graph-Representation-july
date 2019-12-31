import numpy as np
from gensim.models import Word2Vec, KeyedVectors
import random


class DeepWalk(object):

    def __init__(self, args):
        super(DeepWalk, self).__init__()
        self.dimension = args.dimension
        self.walk_length = args.walk_length
        self.walk_num = args.walk_num
        self.window_size = args.window_size
        self.worker = args.worker
        self.iteration = args.iteration

    def train(self, G):
        self.G = G
        walks = self._simulate_walks(self.walk_length, self.walk_num)
        walks = [[str(node) for node in walk] for walk in walks]
        model = Word2Vec(walks, size=self.dimension, window=self.window_size, min_count=0, sg=1, workers=self.worker, iter=self.iteration)
        id2node = dict([(vid, node) for vid, node in enumerate(G.nodes())])
        embeddings = np.asarray([model[str(id2node[i])] for i in range(len(id2node))])
        return embeddings

    def _walk(self, start_node, walk_length):
        # Simulate a random walk starting from start node.
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) == 0:
                break
            k = int(np.floor(np.random.rand() * len(cur_nbrs)))
            walk.append(cur_nbrs[k])
        return walk

    def _simulate_walks(self, walk_length, num_walks):
        # Repeatedly simulate random walks from each node.
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('node number:', len(nodes))
        for walk_iter in range(num_walks):
            print(str(walk_iter + 1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self._walk(node, walk_length))
        return walks
