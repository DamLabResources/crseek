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
    def test_cutoff(self):
        gRNA = 'T' + 'A' * 19
        hitA = 'A' * 20 + 'AGG'
        hitB = 'A'*19 + 'T' + 'CGG'
        hitC = 'T' + 'A' * 19 + 'GGG'

        inp = np.array([[gRNA, hitA],
               [gRNA, hitB],
               [gRNA, hitC]])

        transformer = preprocessing.MatchingTransformer()
        cor = transformer.transform(inp)
        #print(cor.shape)
        mitEst = estimators.MITEstimator()
        mitCut = mitEst.predict(cor)
        #print(mitCut)

        gRNA = 'T' + 'A' * 19
        hitA = 'A' * 20 + 'AGG'
        hitB = 'A' * 19 + 'T' + 'CGG'
        hitC = 'T' + 'A' * 19 + 'CGG'

        inp = np.array([[gRNA, hitA],
                        [gRNA, hitB],
                        [gRNA, hitC]])

        match_encoder = preprocessing.MatchingTransformer()
        matchTmp = match_encoder.transform(inp)
        MITest = estimators.MITEstimator()
        probMIT = MITest.predict(matchTmp)

        cor_prob=[True,False,True]
        
        np.testing.assert_equal(cor_prob[-1], probMIT[-1])

