from crisprtree import preprocessing
from crisprtree import estimators
import pytest
import numpy as np
from itertools import cycle



def make_match_array_from_seqs(grna, seqs):

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


