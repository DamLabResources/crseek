from crisprtree import preprocessing
from crisprtree import estimators
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

    seq_array = np.array(zip(cycle([grna]), seqs))
    return preprocessing.MatchingTransformer().transform(seq_array)


class TestMismatchEstimator(object):

    def test_init(self):

        mod = estimators.MismatchEstimator()

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