from cogdl.datasets import load_npy_data
from cogdl.models import Build_model
from cogdl.datasets import nn_data_utils


class NodeClassification(object):
    """Node classification task."""

    def __init__(self, args):
        self.dataset = args.dataset
        self.X, self.adj, self.labels = load_npy_data(args.data_file, args.dataset)
        self.y_train, self.y_val, self.y_test, self.idx_train, self.idx_val, self.idx_test, self.train_mask = nn_data_utils.get_splits(self.labels)
        self.X /= self.X.sum(1).reshape(-1, 1)
        self.adj = nn_data_utils.preprocess_adj(self.adj, True)
        self.model = Build_model(args).build().build(self.X, self.adj)
        self.max_epoch = args.max_epoch
        self.best_val_loss = 100
        print(self.X, self.adj, self.train_mask)

    def train(self):
        print('start training ...')
        for epoch in range(self.max_epoch):
            self.model.fit([self.X, self.adj], self.y_train, batch_size=self.X.shape[0], sample_weight=self.train_mask, epochs=1, verbose=1, shuffle=False)

            # Predict on full dataset
            preds = self.model.predict([self.X, self.adj], batch_size=self.X.shape[0])

            # Train / validation scores
            train_val_loss, train_val_acc = nn_data_utils.evaluate_preds(preds, [self.y_train, self.y_val],
                                                                         [self.idx_train, self.idx_val])
            print("Epoch: {:04d}".format(epoch),
                  "train_loss= {:.4f}".format(train_val_loss[0]),
                  "train_acc= {:.4f}".format(train_val_acc[0]),
                  "val_loss= {:.4f}".format(train_val_loss[1]),
                  "val_acc= {:.4f}".format(train_val_acc[1]))

            # Early stopping
            if train_val_loss[1] < self.best_val_loss:
                self.best_val_loss = train_val_loss[1]
                wait = 0
            else:
                if wait >= 10:
                    print('Epoch {}: early stopping'.format(epoch))
                    break
                wait += 1
        test_loss, test_acc = nn_data_utils.evaluate_preds(preds, [self.y_test], [self.idx_test])
        result = dict()
        result['dataset'] = self.dataset
        result['train_acc'] = train_val_acc[0]
        result['val_acc'] = train_val_acc[1]
        result['test_acc'] = test_acc
        return result
