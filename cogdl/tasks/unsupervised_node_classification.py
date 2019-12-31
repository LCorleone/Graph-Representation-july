import warnings
from collections import defaultdict

import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import shuffle as skshuffle
from tqdm import tqdm
import pdb


from cogdl.datasets import Build_dataset
from cogdl.models import Build_model


class UnsupervisedNodeClassification(object):
    """Node classification task."""

    def __init__(self, args):
        bd = Build_dataset(root=args.root_dataset)
        self.nodes, self.edge_index, self.label = bd.get_data(name=args.dataset)

        self.num_nodes = self.nodes.shape[0]
        self.num_classes = self.label.shape[1]
        self.label_matrix = self.label

        self.model = Build_model(args).build()

        self.hidden_size = args.hidden_size
        self.num_shuffle = args.num_shuffle

    def train(self):
        print('build the graph from the data ...')
        G = nx.Graph()
        G.add_edges_from([tuple(self.edge_index[:, i]) for i in range(self.edge_index.shape[1])])

        print('model training ...')
        embeddings = self.model.train(G)

        # Map node2id
        features_matrix = np.zeros((self.num_nodes, self.hidden_size))
        for vid, node in enumerate(G.nodes()):
            features_matrix[int(node)] = embeddings[vid]

        # label nor multi-label
        label_matrix = sp.csr_matrix(self.label_matrix)
        return self._evaluate(features_matrix, label_matrix, self.num_shuffle)

    def _evaluate(self, features_matrix, label_matrix, num_shuffle):
        # features_matrix, node2id = utils.load_embeddings(args.emb)
        # label_matrix = utils.load_labels(args.label, node2id, divi_str=" ")

        # shuffle, to create train/test groups
        print('model evaluating ...')
        shuffles = []
        for _ in range(num_shuffle):
            shuffles.append(skshuffle(features_matrix, label_matrix))

        # score each train/test group
        all_results = defaultdict(list)
        # training_percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        training_percents = [0.1, 0.3, 0.5, 0.7, 0.9]

        for train_percent in training_percents:
            for shuf in shuffles:
                X, y = shuf

                training_size = int(train_percent * self.num_nodes)

                X_train = X[:training_size, :]
                y_train = y[:training_size, :]

                X_test = X[training_size:, :]
                y_test = y[training_size:, :]

                clf = TopKRanker(LogisticRegression())
                clf.fit(X_train, y_train)

                # find out how many labels should be predicted
                top_k_list = list(map(int, y_test.sum(axis=1).T.tolist()[0]))
                preds = clf.predict(X_test, top_k_list)
                result = f1_score(y_test, preds, average="micro")
                all_results[train_percent].append(result)
            # print("micro", result)

        return dict(
            (
                f"Micro-F1 {train_percent}",
                sum(all_results[train_percent]) / len(all_results[train_percent]),
            )
            for train_percent in sorted(all_results.keys())
        )


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = sp.lil_matrix(probs.shape)

        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            for label in labels:
                all_labels[i, label] = 1
        return all_labels
