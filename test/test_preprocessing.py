from crisprtree import preprocessing
import pytest
import numpy as np
from itertools import product

from crisprtree.preprocessing import one_hot_encode_row, check_proto_target_input, match_encode_row


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
