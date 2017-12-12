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


class TestMismatchEstimator(object):

    def test_init(self):

        mod = estimators.MismatchEstimator()

    def test_raises_value_error_on_wrong_size(self):

        mod = estimators.MismatchEstimator()
        check = np.ones((5, 20))

        with pytest.raises(ValueError):
            mod.predict(check)


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
                                           miss_tail = 2,
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
                                                           miss_tail = 2,
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
                                           miss_tail = 2,
                                           require_pam = True)
        res = mod.predict(match_array)
        expected = [True, False, False, True, True]
        np.testing.assert_array_equal(res, expected)

        mod = estimators.MismatchEstimator(seed_len = 3,
                                           miss_seed = 0,
                                           miss_tail = 2,
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
                                           miss_tail = 2,
                                           require_pam = True)
        res = mod.predict(match_array)
        expected = [True, True, False, False]
        np.testing.assert_array_equal(res, expected)

        mod = estimators.MismatchEstimator(seed_len = 4,
                                           miss_seed = 2,
                                           miss_tail = 2,
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
                                           miss_tail = 2,
                                           require_pam = True)
        res = mod.predict(match_array)
        expected = [True, True, True, False]
        np.testing.assert_array_equal(res, expected)

        mod = estimators.MismatchEstimator(seed_len = 4,
                                           miss_seed = 0,
                                           miss_tail = 1,
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



class TestCFDEstimator(object):

    def test_raises_value_error_on_wrong_size(self):

        mod = estimators.CFDEstimator()
        check = np.ones((5, 20))

        with pytest.raises(ValueError):
            mod.predict(check)

    def test_loading(self):

        mod = estimators.CFDEstimator()
        assert mod.score_vector.shape == (336, )
        np.testing.assert_approx_equal(mod.score_vector.sum(), 217.9692)


    def test_basic_score(self):
        grna = 'A' * 20
        hits = ['A' * 20 + 'AGG',
                'A'*19 + 'T' + 'CGG',
                'T' + 'A' * 19 + 'GGG',
                'TT' + 'A'*18 + 'GGG',
                'A'*5 + 'TT' + 'A'*13 + 'GGG',
                'A' * 20 + 'GAG',
                ]

        hot_array = make_onehot_array_from_seqs(grna, hits)
        cor_prob = [1.0, 0.6, 1.0, 0.727, 0.714*0.4375, 0.259]

        mod = estimators.CFDEstimator()
        cfd_score = mod.predict_proba(hot_array)

        np.testing.assert_almost_equal(cor_prob, cfd_score, decimal=3)


    def test_cutoff_score(self):

        grna = 'A' * 20
        hits = ['A' * 20 + 'AGG',
                'A'*19 + 'T' + 'CGG',
                'T' + 'A' * 19 + 'GGG',
                'TT' + 'A'*18 + 'GGG',
                'A'*5 + 'TT' + 'A'*13 + 'GGG',
                'A' * 20 + 'GAG',
                ]

        hot_array = make_onehot_array_from_seqs(grna, hits)
        cor_prob = np.array([1.0, 0.6, 1.0, 0.727, 0.714*0.4375, 0.259])

        mod = estimators.CFDEstimator()
        cfd_score = mod.predict(hot_array)

        np.testing.assert_equal(cor_prob>0.75, cfd_score)

        mod = estimators.CFDEstimator(cutoff = 0.5)
        cfd_score = mod.predict(hot_array)
        np.testing.assert_equal(cor_prob>0.5, cfd_score)


    def test_build_pipeline(self):

        grna = 'A' * 20
        hits = ['A' * 20 + 'AGG',
                'A'*19 + 'T' + 'CGG',
                'T' + 'A' * 19 + 'GGG',
                'TT' + 'A'*18 + 'GGG',
                'A'*5 + 'TT' + 'A'*13 + 'GGG',
                'A' * 20 + 'GAG',
                ]

        seq_array = np.array([(grna, hit) for hit in hits])
        cor_prob = np.array([1.0, 0.6, 1.0, 0.727, 0.714*0.4375, 0.259])

        mod = estimators.CFDEstimator.build_pipeline()
        cfd_score = mod.predict_proba(seq_array)

        np.testing.assert_almost_equal(cor_prob, cfd_score, decimal=3)




