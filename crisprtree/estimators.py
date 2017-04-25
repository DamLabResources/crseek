from __future__ import division
from sklearn.base import BaseEstimator
import numpy as np


class MismatchEstimator(BaseEstimator):
    """
    This estimator implements a simple "number of mismatches" determination of
    binding.
    """

    def __init__(self, seed_len=4, miss_seed = 0, miss_non_seed = 3, require_pam = True):
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


class MITEstimator(BaseEstimator):

    def __init__(self, dampening=False, cutoff = 0.75):
        self.dampening=dampening
        self.cutoff = cutoff
        self.penalties = np.array([0, 0, 0.014, 0, 0, 0.395, 0.317, 0,
                          0.389, 0.079, 0.445, 0.508, 0.613,
                          0.851, 0.732, 0.828, 0.615, 0.804,
                          0.685, 0.583])
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        S = self.predict_proba(X)
        return S >= self.cutoff

    def predict_proba(self, X):
        '''
        Parameters
        ----------
        X : np.array

        Returns
        -------
        np.array
            return score array S calculated based on MIT matrix, Nx1 vector

        '''

        if X.shape[1] != 21:
            raise ValueError('Input array shape must be Nx21')

        s1 = (1-(X[:,:-1]==False)*self.penalties).prod(axis=1)

        d = (X[:,:-1]==True).sum(axis=1)
        D = 1/(((19-d)/19)*4 +1)

        mm = (X[:,:-1] == False).sum(axis=1)
        n = mm.copy()
        # there's some hits with zero mismatch, assign to 1 for now
        n[n==0] = 1

        S = s1*D*(1/n**2)
        S[mm==0] = 1
        S *= X[:,-1].astype(float)


        return np.array(S)


def check_matching_input(X):
    """
    Parameters
    ----------
    X : np.array
        Must be an Nx21 of booleans

    Returns
    -------
    bool

    """

    assert X.shape[1] == 21
    #assert np.

