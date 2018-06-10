import numpy as np
import pickle
import logging
import nmslib

x = 1


class NmslibClassifier:

    def __init__(self, space='cosinesimil', method='hnsw'):
        self.space = space
        self.index = nmslib.init(method=method, space=self.space)

    def fit(self, X_train, y_train=None, document_ids=None, ids2class=None, print_progress=False):
        self.index.addDataPointBatch(X_train)
        self.index.createIndex({'post': 2}, print_progress=print_progress)
        # self.classes = y_train if y_train else \
        #    ids2class.columns[ids2class.loc[document_ids].values.argmax(axis=1)]
        self.document_ids = np.array(document_ids)
        self.ids2class = ids2class

    def predict(self, X_test, k=10, weights='uniform', num_threads=-1):
        most_sim = self.index.knnQueryBatch(
            X_test, k=k, num_threads=num_threads)
        most_similar_doc_ids = [self.document_ids[m] for m, dist in most_sim]
        classes = [self.ids2class.loc[m] for m in most_similar_doc_ids]

        if weights == "uniform":
            w = 1
        elif weights == "linear":
            w = np.array(range(k, 0, -1)).reshape(-1, 1)
        elif weights == "hyperbolic":
            w = 1 / (np.array(np.arange(k) + 1)).reshape(-1, 1)
        elif weights == "logarithmic":
            w = np.log(np.arange(1, k + 1) + 1).reshape(-1, 1)

        class_similarity = [(w * c).sum().sort_values(ascending=False)
                            for c in classes]
        logging.debug(class_similarity)
        return np.array([c.index[0] for c in class_similarity])
