from __future__ import division
from crisprtree import preprocessing, estimators
import pytest
import numpy as np
from itertools import cycle



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


class TestMismatchEstimator(object):

    def test_init(self):

        mod = estimators.MismatchEstimator()

    def test_raises_value_error_on_wrong_size(self):

        mod = estimators.MismatchEstimator()
        check = np.ones((5, 20))

        with pytest.raises(ValueError):
            mod.predict(check)

    def test_raises_assert_error_on_too_long_seed(self):

        # Be defensive so we don't have an IndexError on the .predict
        with pytest.raises(AssertionError):
            mod = estimators.MismatchEstimator(seed_len = 21)


    def test_basic_predict(self):

        grna = 'A'*20
        hits = ['A'*20 + 'AGG',          # Perfect hit
                'A'*19 + 'T' + 'AGG',    # One miss in seed
                'T' + 'A'*19 + 'AGG',    # One miss outside seed
                'TTT' + 'A'*17 + 'AGG',  # Three miss outside seed
                'A'*20 + 'ATG'           # No PAM
                ]
        expected = [True, False, True, False, False]

        match_array = make_match_array_from_seqs(grna, hits)

        mod = estimators.MismatchEstimator(seed_len = 4,
                                           miss_seed = 0,
                                           miss_non_seed = 2,
                                           require_pam = True)
        res = mod.predict(match_array)

        np.testing.assert_array_equal(res, expected)

    def test_build_pipeline(self):

        grna = 'A'*20
        hits = ['A'*20 + 'AGG',          # Perfect hit
                'A'*19 + 'T' + 'AGG',    # One miss in seed
                'T' + 'A'*19 + 'AGG',    # One miss outside seed
                'TTT' + 'A'*17 + 'AGG',  # Three miss outside seed
                'A'*20 + 'ATG'           # No PAM
                ]
        expected = [True, False, True, False, False]

        seq_array = np.array(list(zip(cycle([grna]), hits)))

        pipe = estimators.MismatchEstimator.build_pipeline(seed_len = 4,
                                                           miss_seed = 0,
                                                           miss_non_seed = 2,
                                                           require_pam = True)

        res = pipe.predict(seq_array)
        np.testing.assert_array_equal(res, expected)

    def test_set_seed_length(self):

        grna = 'A'*20
        hits = ['A'*20 + 'AGG',
                'A'*19 + 'T' + 'AGG',
                'A'*18 + 'TA' + 'AGG',
                'A'*17 + 'TAA' + 'AGG',
                'A'*16 + 'TAAA' + 'AGG',]

        match_array = make_match_array_from_seqs(grna, hits)

        mod = estimators.MismatchEstimator(seed_len = 2,
                                           miss_seed = 0,
                                           miss_non_seed = 2,
                                           require_pam = True)
        res = mod.predict(match_array)
        expected = [True, False, False, True, True]
        np.testing.assert_array_equal(res, expected)

        mod = estimators.MismatchEstimator(seed_len = 3,
                                           miss_seed = 0,
                                           miss_non_seed = 2,
                                           require_pam = True)
        res = mod.predict(match_array)
        expected = [True, False, False, False, True]
        np.testing.assert_array_equal(res, expected)

    def test_set_seed_mismatch(self):

        grna = 'A'*20
        hits = ['A'*20 + 'AGG',
                'A'*19 + 'T' + 'AGG',
                'A'*18 + 'TT' + 'AGG',
                'A'*17 + 'TTT' + 'AGG',]

        match_array = make_match_array_from_seqs(grna, hits)

        mod = estimators.MismatchEstimator(seed_len = 4,
                                           miss_seed = 1,
                                           miss_non_seed = 2,
                                           require_pam = True)
        res = mod.predict(match_array)
        expected = [True, True, False, False]
        np.testing.assert_array_equal(res, expected)

        mod = estimators.MismatchEstimator(seed_len = 4,
                                           miss_seed = 2,
                                           miss_non_seed = 2,
                                           require_pam = True)
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
                                           miss_non_seed = 2,
                                           require_pam = True)
        res = mod.predict(match_array)
        expected = [True, True, True, False]
        np.testing.assert_array_equal(res, expected)

        mod = estimators.MismatchEstimator(seed_len = 4,
                                           miss_seed = 0,
                                           miss_non_seed = 1,
                                           require_pam = True)
        res = mod.predict(match_array)
        expected = [True, True, False, False]
        np.testing.assert_array_equal(res, expected)




class TestMITestimator(object):

    def test_raises_value_error_on_wrong_size(self):

        mod = estimators.MITEstimator()
        check = np.ones((5, 20))

        with pytest.raises(ValueError):
            mod.predict(check)


    def test_mit_score(self):

        grna = 'A' * 20
        hits = ['A' * 20 + 'AGG',
                'A'*19 + 'T' + 'CGG',
                'T' + 'A' * 19 + 'GGG',
                'TT' + 'A'*18 + 'GGG',
                'A'*5 + 'TT' + 'A'*13 + 'GGG',
                ]

        match_array = make_match_array_from_seqs(grna, hits)

        mit_est = estimators.MITEstimator()
        mit_score = mit_est.predict_proba(match_array)

        cor_prob = [1.0, 0.417, 1, 0.206, 0.0851]

        np.testing.assert_almost_equal(cor_prob, mit_score, decimal=3)

    def test_build_pipeline(self):

        grna = 'A' * 20
        hits = ['A' * 20 + 'AGG',
                'A'*19 + 'T' + 'CGG',
                'T' + 'A' * 19 + 'GGG',
                'TT' + 'A'*18 + 'GGG',
                'A'*5 + 'TT' + 'A'*13 + 'GGG',
                ]

        seq_array = np.array(list(zip(cycle([grna]), hits)))

        pipe = estimators.MITEstimator.build_pipeline()
        mit_score = pipe.predict_proba(seq_array)

        cor_prob = [1.0, 0.417, 1, 0.206, 0.0851]

        np.testing.assert_almost_equal(cor_prob, mit_score, decimal=3)

    def test_cutoff(self):

        grna = 'A' * 20
        hits = ['A' * 20 + 'AGG',
                'A'*19 + 'T' + 'CGG',
                'T' + 'A' * 19 + 'GGG',
                'TT' + 'A' * 18 + 'GGG',
                'A'*5 + 'TT' + 'A'*13 + 'GGG'
                ]

        match_array = make_match_array_from_seqs(grna, hits)
        cor_prob = [True, False, True, False, False]

        mit_est = estimators.MITEstimator(cutoff = 0.75)
        mit_cut = mit_est.predict(match_array)

        np.testing.assert_equal(cor_prob, mit_cut)
        cor_prob = [True, True, True, True, False]

        mit_est = estimators.MITEstimator(cutoff = 0.2)
        mit_cut = mit_est.predict(match_array)

        np.testing.assert_equal(cor_prob, mit_cut)

    def test_requires_pam(self):

        grna = 'T' + 'A' * 19
        hits = ['A' * 20 + 'AGG',
                'A'*19 + 'T' + 'CGG',
                'T' + 'A' * 19 + 'GGG',
                'A' * 20 + 'ATG',]

        match_array = make_match_array_from_seqs(grna, hits)

        mit_est = estimators.MITEstimator(cutoff = 0.75)
        mit_cut = mit_est.predict(match_array)

        cor_prob = [True, False, True, False]

        np.testing.assert_equal(cor_prob, mit_cut)

