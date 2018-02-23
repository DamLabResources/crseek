from __future__ import division
from crisprtree import preprocessing, estimators
import pytest
import numpy as np
from itertools import cycle
from tempfile import NamedTemporaryFile
import os
import yaml
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator



def make_match_array_from_seqs(grna, seqs):
    """
    Utility function for creating a MatchArray (Nx21 boolean) from a list
    of sequences.

    Parameters
    ----------
    grna : str
        A 20 bp gRNA
    seqs : iter
        An iterable of 23bp target sequences

    Returns
    -------
    np.array

    """

    seq_array = np.array(list(zip(cycle([grna]), seqs)))
    return preprocessing.MatchingTransformer().transform(seq_array)

def make_onehot_array_from_seqs(grna, seqs):
    """
    Utility function for creating a OneHotArray (Nx336 boolean) from a list
    of sequences.

    Parameters
    ----------
    grna : str
        A 20 bp gRNA
    seqs : iter
        An iterable of 23bp target sequences

    Returns
    -------
    np.array

    """

    seq_array = np.array(list(zip(cycle([grna]), seqs)))
    return preprocessing.OneHotTransformer().transform(seq_array)



class BaseChecker(object):

    hits = ['A'*20 + 'AGG',          # Perfect hit
            'A'*19 + 'T' + 'AGG',    # One miss in seed
            'T' + 'A'*19 + 'AGG',    # One miss outside seed
            'TTT' + 'A'*17 + 'AGG',  # Three miss outside seed
            'A'*20 + 'ATG'           # No PAM
            ]
    expected = [0, 0, 0, 0, 0]
    prob = None

    estimator = None
    pipeline = None

    def _make_match_array(self, grna, hits):
        return make_match_array_from_seqs(grna, hits)

    def test_raises_value_error_on_wrong_size(self):

        check = np.ones((5, 20))

        with pytest.raises(ValueError):
            self.estimator.predict(check)

    def test_basic_predict(self):
        grna = 'A'*20

        if self.estimator is None:
            raise NotImplementedError

        match_array = self._make_match_array(grna, self.hits)

        res = self.estimator.predict(match_array)
        np.testing.assert_array_equal(res, self.expected)

    def test_predict_proba(self):

        if self.prob is not None:
            grna = 'A'*20

            if self.estimator is None:
                raise NotImplementedError

            match_array = self._make_match_array(grna, self.hits)

            res = self.estimator.predict_proba(match_array)
            np.testing.assert_almost_equal(res, self.prob, decimal=3)

    def test_change_cutoff(self):

        if self.prob is not None:
            grna = 'A'*20

            if self.estimator is None:
                raise NotImplementedError

            match_array = self._make_match_array(grna, self.hits)

            self.estimator.cutoff = 0.2
            res = self.estimator.predict(match_array)
            np.testing.assert_array_equal(res, np.array(self.prob)>0.2)

            self.estimator.cutoff = 0.5
            res = self.estimator.predict(match_array)
            np.testing.assert_array_equal(res, np.array(self.prob)>0.5)

            self.estimator.cutoff = 0.8
            res = self.estimator.predict(match_array)
            np.testing.assert_array_equal(res, np.array(self.prob)>0.8)

    def test_build_pipeline(self):
        grna = 'A'*20

        if self.pipeline is None:
            raise NotImplementedError

        seq_array = np.array(list(zip(cycle([grna]), self.hits)))

        res = self.pipeline.predict(seq_array)
        np.testing.assert_array_equal(res, self.expected)


class TestMismatchEstimator(BaseChecker):

    hits = ['A'*20 + 'AGG',          # Perfect hit
            'A'*19 + 'T' + 'AGG',    # One miss in seed
            'T' + 'A'*19 + 'AGG',    # One miss outside seed
            'TTT' + 'A'*17 + 'AGG',  # Three miss outside seed
            'A'*20 + 'ATG'           # No PAM
            ]
    expected = [True, False, True, False, False]
    prob = None

    estimator = estimators.MismatchEstimator(seed_len = 4, miss_seed = 0,
                                             miss_tail = 2, pam = 'NGG')
    pipeline = estimators.MismatchEstimator.build_pipeline(seed_len = 4,
                                                           miss_seed = 0,
                                                           miss_tail = 2,
                                                           pam = 'NGG')
    def test_init(self):

        mod = estimators.MismatchEstimator()

    def test_check_pipeline(self):

        assert self.pipeline.matcher.seed_len == 4
        assert self.pipeline.matcher.miss_seed == 0
        assert self.pipeline.matcher.miss_tail == 2
        assert self.pipeline.matcher.pam == 'NGG'

    def test_load_yaml(self):

        d = {'Seed Length': 5,
             'Tail Length': 15,
             'Seed Misses': 1,
             'Tail Misses': 3,
             'PAM': 'NRG'}

        with NamedTemporaryFile(suffix='.yaml', mode='w') as handle:
            yaml.dump(d, handle)
            handle.flush()
            os.fsync(handle)

            est = estimators.MismatchEstimator.load_yaml(handle.name)

        assert type(est) is Pipeline
        mm_est = est.steps[1][1]

        assert mm_est.seed_len == 5
        assert mm_est.tail_len == 15
        assert mm_est.miss_seed == 1
        assert mm_est.miss_tail == 3
        assert mm_est.pam == 'NRG'

    def test_set_seed_length(self):

        grna = 'A'*20
        hits = ['A'*20 + 'AGG',
                'A'*19 + 'T' + 'AGG',
                'A'*18 + 'TA' + 'AGG',
                'A'*17 + 'TAA' + 'AGG',
                'A'*16 + 'TAAA' + 'AGG',]

        match_array = self._make_match_array(grna, hits)

        mod = estimators.MismatchEstimator(seed_len = 2,
                                           miss_seed = 0,
                                           miss_tail = 2,
                                           pam = 'NGG')
        res = mod.predict(match_array)
        expected = [True, False, False, True, True]
        np.testing.assert_array_equal(res, expected)

        mod = estimators.MismatchEstimator(seed_len = 3,
                                           miss_seed = 0,
                                           miss_tail = 2,
                                           pam = 'NGG')
        res = mod.predict(match_array)
        expected = [True, False, False, False, True]
        np.testing.assert_array_equal(res, expected)

    def test_set_seed_mismatch(self):

        grna = 'A'*20
        hits = ['A'*20 + 'AGG',
                'A'*19 + 'T' + 'AGG',
                'A'*18 + 'TT' + 'AGG',
                'A'*17 + 'TTT' + 'AGG']

        match_array = self._make_match_array(grna, hits)

        mod = estimators.MismatchEstimator(seed_len = 4,
                                           miss_seed = 1,
                                           miss_tail = 2,
                                           pam = 'NGG')
        res = mod.predict(match_array)
        expected = [True, True, False, False]
        np.testing.assert_array_equal(res, expected)

        mod = estimators.MismatchEstimator(seed_len = 4,
                                           miss_seed = 2,
                                           miss_tail = 2,
                                           pam = 'NGG')
        res = mod.predict(match_array)
        expected = [True, True, True, False]
        np.testing.assert_array_equal(res, expected)

    def test_set_non_seed_mismatch(self):

        grna = 'A'*20
        hits = ['A'*20 + 'AGG',
                'T' + 'A'*19 + 'AGG',
                'TT' + 'A'*18 + 'AGG',
                'TTT' + 'A'*17 + 'AGG']

        match_array = make_match_array_from_seqs(grna, hits)

        mod = estimators.MismatchEstimator(seed_len = 4,
                                           miss_seed = 0,
                                           miss_tail = 2,
                                           pam = 'NGG')
        res = mod.predict(match_array)
        expected = [True, True, True, False]
        np.testing.assert_array_equal(res, expected)

        mod = estimators.MismatchEstimator(seed_len = 4,
                                           miss_seed = 0,
                                           miss_tail = 1,
                                           pam = 'NGG')
        res = mod.predict(match_array)
        expected = [True, True, False, False]
        np.testing.assert_array_equal(res, expected)


class TestMITestimator(BaseChecker):

    hits = ['A' * 20 + 'AGG',
            'A'*19 + 'T' + 'CGG',
            'T' + 'A' * 19 + 'GGG',
            'TT' + 'A'*18 + 'GGG',
            'A'*5 + 'TT' + 'A'*13 + 'GGG']
    expected = [True, False, True, True, False]
    prob = [1.0, 0.417, 1, 1, 0.413]
    estimator = estimators.MITEstimator()
    pipeline = estimators.MITEstimator.build_pipeline()

    def test_requires_pam(self):

        grna = 'T' + 'A' * 19
        hits = ['A' * 20 + 'AGG',
                'A'*19 + 'T' + 'CGG',
                'T' + 'A' * 19 + 'GGG',
                'A' * 20 + 'ATG',]

        match_array = make_match_array_from_seqs(grna, hits)

        mit_est = estimators.MITEstimator(cutoff = 0.05)
        mit_cut = mit_est.predict(match_array)

        cor_prob = [True, True, True, False]

        np.testing.assert_equal(cor_prob, mit_cut)


class TestKineticEstimator(BaseChecker):

    hits = ['A' * 20 + 'AGG',
            'A'*19 + 'T' + 'CGG',
            'T' + 'A' * 19 + 'GGG',
            'TT' + 'A'*18 + 'GGG',
            'A'*5 + 'TT' + 'A'*13 + 'GGG']
    expected = [True, False, True, True, True]
    prob = [1.0, 1.12701e-8, 0.7399, 0.5475, 0.5447311]

    estimator = estimators.KineticEstimator()
    pipeline = estimators.KineticEstimator.build_pipeline()

    def test_requires_pam(self):

        grna = 'A' * 20
        hits = ['A' * 20 + 'AGG',
                'A'*19 + 'T' + 'CGG',
                'T' + 'A' * 19 + 'GGG',
                'A' * 20 + 'ATG',]

        match_array = make_match_array_from_seqs(grna, hits)

        est = estimators.KineticEstimator(cutoff = 0.50)
        cut = est.predict(match_array)

        cor_prob = [True, False, True, False]

        np.testing.assert_equal(cor_prob, cut)


class TestCFDEstimator(BaseChecker):

    hits = ['A' * 20 + 'AGG',
                'A'*19 + 'T' + 'CGG',
                'T' + 'A' * 19 + 'GGG',
                'TT' + 'A'*18 + 'GGG',
                'A'*5 + 'TT' + 'A'*13 + 'GGG',
                'A' * 20 + 'GAG']
    expected = [True, False, True, False, False, False]
    prob = [1.0, 0.6, 1.0, 0.727, 0.714*0.4375, 0.259]

    estimator = estimators.CFDEstimator()
    pipeline = estimators.CFDEstimator.build_pipeline()

    def _make_match_array(self, grna, hits):
        return make_onehot_array_from_seqs(grna, hits)

    def test_loading(self):

        mod = estimators.CFDEstimator()
        assert mod.score_vector.shape == (336, )
        np.testing.assert_approx_equal(mod.score_vector.sum(), 217.9692)






