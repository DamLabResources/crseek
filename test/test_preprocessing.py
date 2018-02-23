from crisprtree import preprocessing
import pytest
import numpy as np
import pandas as pd
from itertools import product

from crisprtree.preprocessing import one_hot_encode_row, check_proto_target_input, match_encode_row, locate_hits_in_array
from crisprtree.estimators import CFDEstimator
from crisprtree import utils


class TestBasicInputs(object):

    def test_simple_inputs(self):

        gRNA = 'A'*20
        hit = 'A'*20 + 'AGG'

        inp = np.array([[gRNA, hit],
                         [gRNA, hit]])

        assert check_proto_target_input(inp)

    def test_missing_col(self):

        gRNA = 'A'*20
        hit = 'A'*20 + 'AGG'

        inp = np.array([[gRNA],
                         [gRNA]])

        checks = [check_proto_target_input,
                  preprocessing.MatchingTransformer().transform,
                  preprocessing.OneHotTransformer().transform
                  ]

        for check in checks:
            with pytest.raises(AssertionError):
                check(inp)

    def test_missing_PAM(self):

        gRNA = 'A'*20
        hit = 'A'*20

        inp = np.array([[gRNA, hit],
                         [gRNA, hit]])

        checks = [check_proto_target_input,
                  preprocessing.MatchingTransformer().transform,
                  preprocessing.OneHotTransformer().transform
                  ]

        for check in checks:
            with pytest.raises(AssertionError):
                check(inp)


class TestOneHotEncoding(object):

    def test_encoding(self):

        gRNA = 'A'*20
        hit = 'A'*20 + 'AGG'

        cor = np.zeros(21*16)
        locs = np.arange(0, 20*16, 16)
        cor[locs] = True
        cor[-6] = True # GG

        res = one_hot_encode_row(gRNA, hit)
        assert res.shape == (21*16, )

        np.testing.assert_array_equal(cor.astype(bool), res)

    def test_more_encoding(self):

        gRNA = 'T' + 'A'*19
        hit = 'A'*20 + 'AGG'

        cor = np.zeros(21*16)
        locs = np.arange(0, 20*16, 16)
        cor[locs] = True
        cor[0] = False
        cor[12] = True
        cor[-6] = True # GG

        res = one_hot_encode_row(gRNA, hit)

        assert res.shape == (21*16, )

        np.testing.assert_array_equal(cor.astype(bool), res)

    def test_PAM_encoding(self):

        gRNA = 'T' + 'A'*19

        locs = np.arange(0, 20*16, 16)

        pams = product('ACGT', repeat=2)
        for pos, (p1, p2) in enumerate(pams):
            cor = np.zeros(21*16)
            cor[locs] = True
            cor[0] = False
            cor[12] = True
            cor[20*16+pos] = True # PAM
            hit = 'A'*20 + 'A' + p1 + p2

            res = one_hot_encode_row(gRNA, hit)

            np.testing.assert_array_equal(cor.astype(bool), res)

    def test_transforming(self):

        gRNA = 'T' + 'A'*19
        hit = 'A'*20 + 'AGG'

        cor = np.zeros(21*16)
        locs = np.arange(0, 20*16, 16)
        cor[locs] = True
        cor[0] = False
        cor[12] = True
        cor[-6] = True # GG

        inp = np.array([[gRNA, hit],
                        [gRNA, hit],
                        [gRNA, hit]])
        hot_encoder = preprocessing.OneHotTransformer()
        res = hot_encoder.transform(inp)

        assert res.shape == (3, 21*16)

        for row in range(3):
            np.testing.assert_array_equal(cor.astype(bool), res[row, :])

    def test_bad_target(self):

        gRNA = 'A'*20
        hit = 'N' + 'A'*19 + 'AGG'

        with pytest.raises(AssertionError):
            preprocessing.one_hot_encode_row(gRNA, hit)


class TestMatchingEncoding(object):

    def test_encoding(self):

        gRNA = 'A'*20
        hit = 'A'*20 + 'AGG'
        cor = np.array([True]*21)

        res = match_encode_row(gRNA, hit)
        assert res.shape == (20+1, )

        np.testing.assert_array_equal(cor.astype(bool), res)

    def test_more_encoding(self):

        gRNA = 'T' + 'A'*19
        hit = 'A'*20 + 'AGG'

        cor = np.array([False] + [True]*20)

        res = match_encode_row(gRNA, hit)

        assert res.shape == (20+1, )

        np.testing.assert_array_equal(cor.astype(bool), res)

    def test_transforming(self):

        gRNA = 'T' + 'A'*19
        hit = 'A'*20 + 'AGG'

        cor = np.array([False] + [True]*20)

        inp = np.array([[gRNA, hit],
                        [gRNA, hit],
                        [gRNA, hit]])
        hot_encoder = preprocessing.MatchingTransformer()
        res = hot_encoder.transform(inp)

        assert res.shape == (3, 20+1)

        for row in range(3):
            np.testing.assert_array_equal(cor.astype(bool), res[row,:])


def make_random_seq(bp):
    """ Utility function for making random sequence
    Parameters
    ----------
    bp : int
        Length of sequence

    Returns
    -------
    str

    """
    return ''.join(np.random.choice(list('ACGT'), size = bp))


@pytest.mark.skipif(utils._missing_casoffinder(), reason="Need CasOff installed")
class TestLocate(object):

    def test_basic(self):

        #prevent Heisenbugs
        np.random.seed(0)

        seqs = [make_random_seq(50) + 'TTTT' + 'A'*20 + 'CGG' + 'TTTT' + make_random_seq(50),
                make_random_seq(12) + 'TTTT' + 'C'*19+'T' + 'CGG' + 'TTTT' + make_random_seq(50),
                make_random_seq(75),
                make_random_seq(25) + 'TTTT' + 'T' + 'A'*19 + 'TGG' + 'TTTT' + make_random_seq(50)]

        grnas = ['A'*20,
                 'C'*19+'T',
                 'C'*19+'T',
                 'A'*20]

        X = np.array(list(zip(grnas, seqs)))
        estimator = CFDEstimator.build_pipeline()

        nX, loc = locate_hits_in_array(X, estimator, mismatches=6)

        assert nX.shape == (4, 2)
        assert loc.shape == (4, 2)

        np.testing.assert_array_equal([54, 16, np.nan, 29], loc[:,0])
        np.testing.assert_array_equal([1, 1, np.nan, 1], loc[:,1])

        cor_hit = ['A'*20 + 'CGG', 'C'*19+'T' + 'CGG', np.nan, 'T' + 'A'*19 + 'TGG']
        cor_grnas = ['A'*20, 'C'*19+'T', np.nan, 'A'*20]
        cX = pd.DataFrame(list(zip(cor_grnas, cor_hit))).values

        mask = np.array([True, True, False, True])
        np.testing.assert_array_equal(cX[mask, :], nX[mask, :])

        assert np.isnan(cX[2, 0])
        assert np.isnan(cX[2, 1])
