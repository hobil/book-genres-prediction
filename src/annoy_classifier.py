#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 14:34:15 2018.

@author: j.bilek
"""

import numpy as np
import pickle
from annoy import AnnoyIndex


class AbstractAnnoyClassifier:
    def __init__(self):
        pass

    @classmethod
    def load(cls, filename):
        document_ids, ids2class, n_trees, dim = pickle.load(open(filename, 'rb'))
        aac = AbstractAnnoyClassifier()
        aac.document_ids = document_ids
        aac.ids2class = ids2class
        aac.n_trees = n_trees
        aac.dim = dim
        return aac


class AnnoyClassifier(AbstractAnnoyClassifier):
    """Does not store the doc2vec model. Expects already vectors in X_test."""

    def __init__(self, doc2vec_model, document_ids, ids2class, n_trees=100):
        """[summary].

        Arguments:
            doc2vec_model {[type]} -- [description]
            document_ids {[type]} -- [description]
            ids2class {[type]} -- [description]

        Keyword Arguments:
            n_trees {int} -- [description] (default: {100})
        """
        self.document_ids = document_ids
        self.ids2class = ids2class
        self.n_trees = n_trees

        self.dim = doc2vec_model.vector_size
        self.annoy_index = AnnoyIndex(self.dim)
        # for some reason there is a KeyError at the end of the enumeration
        try:
            for i, vec in enumerate(doc2vec_model.docvecs):
                self.annoy_index.add_item(i, np.array(vec))
        except KeyError as e:
            print(e)
        self.annoy_index.build(self.n_trees)

    @classmethod
    def load(cls, filename):
        """[summary].

        Arguments:
            filename {[type]} -- [description]
        """
        annoy_filename = filename + '.ann'
        attr_filename = filename + '.pkl'
        ac = super(AnnoyClassifier, cls).load(attr_filename)
        ac.__class__ = AnnoyClassifier
        ac.annoy_index = AnnoyIndex(ac.dim)
        ac.annoy_index.load(annoy_filename)
        return ac

    def save(self, filename):
        """[summary].

        Arguments:
            filename {[type]} -- [description]
        """
        annoy_filename = filename + '.ann'
        attr_filename = filename + '.pkl'
        attr = (self.document_ids, self.ids2class, self.n_trees, self.dim)
        self.annoy_index.save(annoy_filename)
        pickle.dump(attr, open(attr_filename, 'wb'))

    def single_predict(self, vec, n_nearest):
        most_sim_ind = self.annoy_index.get_nns_by_vector(vec, n_nearest)
        most_similar_doc_ids = [self.document_ids[x] for x in most_sim_ind]
        return self.ids2class.loc[most_similar_doc_ids].sum().\
            sort_values(ascending=False).index[0]

    def single_predict_proba(self, vec, n_nearest):
        most_sim_ind = self.annoy_index.get_nns_by_vector(vec, n_nearest)
        most_similar_doc_ids = [self.document_ids[x] for x in most_sim_ind]
        return self.ids2class.loc[most_similar_doc_ids].mean().\
            sort_values(ascending=False)

    def predict(self, X_test, n_nearest=10):
        """[summary].
            TODO:
             - take ranking?
             - take distance?
             - what n_nearest?

        Arguments:
            X_test {[type]} -- [description]

        Keyword Arguments:
            n_nearest {int} -- [description] (default: {10})

        Returns:
            [type] -- [description]

        """
        return [self.single_predict(v, n_nearest) for v in X_test]

    def predict_probas(self, X_test, n_nearest=10):
        """[summary].
            TODO:
             - take ranking?
             - take distance?
             - what n_nearest?

        Arguments:
            X_test {[type]} -- [description]

        Keyword Arguments:
            n_nearest {int} -- [description] (default: {10})

        Returns:
            [type] -- [description]

        """
        return [self.single_predict_proba(v, n_nearest) for v in X_test]
