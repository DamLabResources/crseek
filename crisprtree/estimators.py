from __future__ import division
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from crisprtree.preprocessing import MatchingTransformer, OneHotTransformer
import numpy as np
import os

this_dir, this_filename = os.path.split(os.path.abspath(__file__))
DATA_PATH = os.path.join(this_dir, '..',  "data")


class MismatchEstimator(BaseEstimator):
    """
    This estimator implements a simple "number of mismatches" determination of
    binding.
    """

    def __init__(self, seed_len = 4, miss_seed = 0, miss_non_seed = 3, require_pam = True):
        """

        Parameters
        ----------
        seed_len : int
            The length of the seed region.
        miss_seed : int
            The number of mismatches allowed in the seed region.
        miss_non_seed : int
            The number of mismatches allowed in the non-seed region.
        require_pam : bool
            Must the PAM be present

        Returns
        -------
        MismatchEstimator
        """

        assert seed_len <= 20, 'seed_len cannot be longer then 20'
        self.seed_len = seed_len
        self.miss_seed = miss_seed
        self.miss_non_seed = miss_non_seed
        self.require_pam = require_pam

    @staticmethod
    def build_pipeline(**kwargs):
        """ Utility function to build a pipeline.
        Parameters
        ----------
        Keyword arguements are passed to the Estimator on __init__

        Returns
        -------

        Pipeline

        """

        pipe = Pipeline(steps = [('transform', MatchingTransformer()),
                                 ('predict', MismatchEstimator(**kwargs))])
        return pipe

    def fit(self, X, y = None):
        return self

    def predict(self, X):
        """

        Parameters
        ----------
        X : array
            Should be Nx21 as produced by preprocessing.MatchingTransformer
        Returns
        -------

        """

        if X.shape[1] != 21:
            raise ValueError('Input array shape must be Nx21')

        seed_miss = (X[:, -(self.seed_len+1):-1] == False).sum(axis=1)
        non_seed_miss = (X[:, :-(self.seed_len)] == False).sum(axis=1)

        binders = (seed_miss <= self.miss_seed) & (non_seed_miss <= self.miss_non_seed)
        if self.require_pam:
            binders &= X[:, -1]

        return binders

    def predict_proba(self, X):
        return self.predict(X)


class MITEstimator(BaseEstimator):

    def __init__(self, dampen = True, cutoff = 0.75):
        """
        Parameters
        ----------
        cutoff : float
            Cutoff for calling binding

        Returns
        -------

        MITEstimator

        """
        self.cutoff = cutoff
        self.dampen = dampen
        self.penalties = np.array([0, 0, 0.014, 0, 0, 0.395, 0.317, 0,
                                   0.389, 0.079, 0.445, 0.508, 0.613,
                                   0.851, 0.732, 0.828, 0.615, 0.804,
                                   0.685, 0.583])

    @staticmethod
    def build_pipeline(**kwargs):
        """ Utility function to build a pipeline.
        Parameters
        ----------
        Keyword arguements are passed to the Estimator on __init__

        Returns
        -------

        Pipeline

        """

        pipe = Pipeline(steps = [('transform', MatchingTransformer()),
                                 ('predict', MITEstimator(**kwargs))])
        return pipe

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        S = self.predict_proba(X)
        return S >= self.cutoff

    def predict_proba(self, X):
        """
        Parameters
        ----------
        X : np.array

        Returns
        -------
        np.array
            return score array S calculated based on MIT matrix, Nx1 vector

        """

        if X.shape[1] != 21:
            raise ValueError('Input array shape must be Nx21')

        s1 = (1-(X[:,:-1] == np.array([False]))*self.penalties).prod(axis=1)

        mm = (X[:, :-1] == np.array([False])).sum(axis=1)
        n = mm.copy()

        def distance(x):
            idx = np.where(x==False)
            if len(idx[0]) > 1:
                return (idx[0][-1] - idx[0][0])
            else:
                return 0

        d = np.apply_along_axis(distance, axis=1, arr=X[: , :-1])
        with np.errstate(divide='ignore', invalid='ignore'):
            d = np.true_divide(d,(n-1))
            d[d == np.inf] = 0
            d = np.nan_to_num(d)


        D = 1 / ((19-d) / 19 * 4 + 1)
        D[n<2] =1
        S = s1
        psudoN = n.copy()
        psudoN[n <1] =1
        if self.dampen:
            S = s1*D*(np.array([1])/psudoN**2)

        S[mm==0]=1
        S *= X[:, -1].astype(float)

        return np.array(S)


class CFDEstimator(BaseEstimator):

    def __init__(self, cutoff = 0.75):
        """
        Parameters
        ----------
        cutoff : float
            Cutoff for calling binding

        Returns
        -------

        CFDEstimator

        """

        self.cutoff = cutoff
        self._read_scores()

    def _read_scores(self):

        with open(os.path.join(DATA_PATH, 'cfdMatrix.csv')) as handle:
            vals = []
            for line in handle:
                line = line.replace(',', '').replace('[', '').replace(']', '').strip()
                vals += [float(v) for v in line.split()]

        self.score_vector = np.array(vals)


    @staticmethod
    def build_pipeline(**kwargs):
        """ Utility function to build a pipeline.
        Parameters
        ----------
        Keyword arguements are passed to the Estimator on __init__

        Returns
        -------

        Pipeline

        """

        pipe = Pipeline(steps = [('transform', OneHotTransformer()),
                                 ('predict', CFDEstimator(**kwargs))])
        return pipe


    def fit(self, X, y=None):
        return self


    def predict(self, X):

        return self.predict_proba(X) >= self.cutoff


    def predict_proba(self, X):

        if X.shape[1] != 336:
            raise ValueError('Input array shape must be Nx336')

        items = X.shape[0]
        scores = np.tile(self.score_vector, (items, 1))
        hot_scores = scores[X].reshape(-1, 21)

        probs = np.prod(hot_scores, axis=1)
        return probs



