from __future__ import division
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from crisprtree.preprocessing import MatchingTransformer, OneHotTransformer
import numpy as np
import os
import yaml


this_dir, this_filename = os.path.split(os.path.abspath(__file__))
DATA_PATH = os.path.join(this_dir, '..',  "data")


class MismatchEstimator(BaseEstimator):
    """
    This estimator implements a simple "number of mismatches" determination of
    binding.
    """

    def __init__(self, seed_len = 4, tail_len = 16, miss_seed = 0, miss_tail = 3, pam = 'NGG'):
        """

        Parameters
        ----------
        seed_len : int
            The length of the seed region.
        tail_len : int
            The length of the tail region.
        miss_seed : int
            The number of mismatches allowed in the seed region.
        miss_tail : int
            The number of mismatches allowed in the tail region.
        pam : str
            Must the PAM be present

        Returns
        -------
        MismatchEstimator
        """

        self.seed_len = seed_len
        self.tail_len = tail_len
        self.miss_seed = miss_seed
        self.miss_tail = miss_tail
        self.pam = pam

    @staticmethod
    def load_yaml(path):

        with open(path) as handle:
            data = yaml.load(handle)

        kwargs = {'seed_len': data.get('Seed Length', 4),
                  'tail_len': data.get('Tail Length', 16),
                  'miss_seed': data.get('Seed Misses', 0),
                  'miss_tail': data.get('Tail Misses', 3),
                  'pam': data.get('PAM', 'NGG')}

        return MismatchEstimator.build_pipeline(**kwargs)

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
        pipe.matcher = pipe.steps[1][1]

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

        binders = (seed_miss <= self.miss_seed) & (non_seed_miss <= self.miss_tail)
        if self.pam:
            binders &= X[:, -1]

        return binders

    def predict_proba(self, X):
        return self.predict(X)


class MITEstimator(BaseEstimator):

    def __init__(self, cutoff = 0.75):
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

        d = (X[:,:-1] == np.array([True])).sum(axis=1)
        D = 1/(((19-d)/19)*4 +1)

        mm = (X[:,:-1] == np.array([False])).sum(axis=1)
        n = mm.copy()
        # there's some hits with zero mismatch, assign to 1 for now
        n[n==0] = 1

        S = s1*D*(np.array([1])/n**2)
        S[mm==0] = 1
        S *= X[:,-1].astype(float)

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



class KineticEstimator(BaseEstimator):

    def __init__(self, nseed = 11, dC = 1.8, Pmax = 0.74, variant = None,
                 dI = None, npair = None, pam = 'NGG', cutoff = 0.5):
        """
        Parameters
        ----------
        nseed : int
            Position at which half of single-missmatches will fail to cleave their target
        dC : float
            Free energy associated with a correct binding pair.
        Pmax : float
            The maximal cleavage efficiency of any single missmatch target
        variant : str
            Name of variant described in the paper. Must be:
             {spCas9,LbCpf1, AsCpf1}
             Sets all appropriate constants
        dI : float
            Free energy gain associated with an incorrect binding pair.
            Currently unused.
        npair : int
            Allowable distance between missmathes. Currently unused.
        pam : str
            PAM sequence.
        cutoff : float
            Probability cutoff to use when making binary comparisons.

        Returns
        -------

        KineticEstimator

        """

        if variant is not None:
            # Values taken from Figure 6: https://doi.org/10.1016/j.celrep.2018.01.045
            # Klein et al, Cell Reports, Feb 2018
            knowns = {'spCas9': {'nseed': 11, 'dC': 1.8, 'Pmax': 0.74, 'pam': 'NGG'},
                      'LbCpf1': {'nseed': 19, 'dC': 2.1, 'Pmax': 0.83, 'pam': 'NGG'},
                      'AsCpf1': {'nseed': 19, 'dC': 4, 'Pmax': 0.83, 'pam': 'NGG'}}
            try:
                data = knowns[variant]
            except KeyError:
                msg = 'Variant must be one of {spCas9,LbCpf1, AsCpf1} got: %s' % variant
                raise ValueError(msg)

            self.nseed = data['nseed']
            self.dC = data['dC']
            self.Pmax = data['Pmax']
            self.pam = data['pam']

        else:
            self.nseed = nseed
            self.dC = dC
            self.Pmax = Pmax
            self.pam = pam

        self.cutoff = cutoff


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
                                 ('predict', KineticEstimator(**kwargs))])
        return pipe

    def fit(self, X, y=None):
        return self

    def predict(self, X):

        return self.predict_proba(X) >= self.cutoff

    def predict_proba(self, X):

        if X.shape[1] != 21:
            raise ValueError('Input array shape must be Nx21')

        # Input array  : Position 0 => PAM distal region
        # Klein model  : Position 0 => PAM proximal region

        pos = np.arange(20, 0, -1)
        pclv = self.Pmax/(1+np.exp(-(pos-self.nseed)*self.dC))

        vals = pclv*(X[:,:-1]==False)

        scores = np.ones_like(vals)
        scores[X[:,:-1]==False] = vals[X[:,:-1]==False]

        probs = scores.prod(axis=1)

        # PAMs must be present
        probs *= X[:, -1]
        return probs
