from crisprtree import preprocessing
import pytest
import numpy as np

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

        cor = np.zeros(20*16)
        locs = np.arange(0, 20*16, 16)
        cor[locs] = True

        res = one_hot_encode_row(gRNA, hit)
        assert res.shape == (20*16+1, )

        np.testing.assert_array_equal(cor.astype(bool), res[:20*16])
        assert res[-1] == True

    def test_more_encoding(self):

        gRNA = 'T' + 'A'*19
        hit = 'A'*20 + 'AGG'

        cor = np.zeros(20*16)
        locs = np.arange(0, 20*16, 16)
        cor[locs] = True
        cor[0] = False
        cor[12] = True

        res = one_hot_encode_row(gRNA, hit)

        assert res.shape == (20*16+1, )

        np.testing.assert_array_equal(cor.astype(bool), res[:20*16])
        assert res[-1] == True

    def test_transforming(self):

        gRNA = 'T' + 'A'*19
        hit = 'A'*20 + 'AGG'

        cor = np.zeros(20*16)
        locs = np.arange(0, 20*16, 16)
        cor[locs] = True
        cor[0] = False
        cor[12] = True

        inp = np.array([[gRNA, hit],
                        [gRNA, hit],
                        [gRNA, hit]])
        hot_encoder = preprocessing.OneHotTransformer()
        res = hot_encoder.transform(inp)

        assert res.shape == (3, 20*16+1)

        for row in range(3):
            np.testing.assert_array_equal(cor.astype(bool), res[row,:20*16])
            assert res[row,-1] == True


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
