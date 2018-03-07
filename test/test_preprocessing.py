from itertools import product
from functools import partial

import numpy as np
import pandas as pd
import pytest
from Bio.Alphabet import generic_dna, generic_rna, IUPAC
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from crisprtree import exceptions
from crisprtree import preprocessing
from crisprtree import utils
from crisprtree.estimators import CFDEstimator


class TestBasicInputs(object):
    def test_simple_inputs(self):

        spacer = Seq('A' * 20, alphabet=generic_rna)
        target = Seq('A' * 20 + 'AGG', alphabet=generic_dna)

        inp = np.array([[spacer, target],
                        [spacer, target]])

        assert preprocessing.check_proto_target_input(inp)

    def test_str_inputs(self):

        spacer = 'A' * 20
        target = 'A' * 20 + 'AGG'

        inp = np.array([[spacer, target],
                        [spacer, target]])

        with pytest.raises(ValueError):
            preprocessing.check_proto_target_input(inp)

    def test_bad_alphabet_inputs(self):

        spacer_d = Seq('A' * 20, alphabet=generic_dna)
        spacer_r = Seq('A' * 20, alphabet=generic_rna)
        target_d = Seq('A' * 20 + 'CGG', alphabet=generic_dna)
        target_r = Seq('A' * 20 + 'CGG', alphabet=generic_rna)

        checks = [preprocessing.check_proto_target_input,
                  preprocessing.MatchingTransformer().transform,
                  preprocessing.OneHotTransformer().transform
                  ]

        inp = np.array([[spacer_r, target_r],
                        [spacer_r, target_r]])

        for check in checks:
            with pytest.raises(exceptions.WrongAlphabetException):
                check(inp)

        inp = np.array([[spacer_d, target_d],
                        [spacer_d, target_d]])

        for check in checks:
            with pytest.raises(exceptions.WrongAlphabetException):
                check(inp)

    def test_missing_col(self):

        spacer = Seq('A' * 20, alphabet=generic_rna)

        inp = np.array([[spacer],
                        [spacer]])

        checks = [preprocessing.check_proto_target_input,
                  preprocessing.MatchingTransformer().transform,
                  preprocessing.OneHotTransformer().transform
                  ]

        for check in checks:
            with pytest.raises(AssertionError):
                check(inp)

    def test_missing_PAM(self):

        spacer = Seq('A' * 20, alphabet=generic_rna)
        target = Seq('A' * 20, alphabet=generic_dna)

        inp = np.array([[spacer, target],
                        [spacer, target]])

        checks = [preprocessing.check_proto_target_input,
                  preprocessing.MatchingTransformer().transform,
                  preprocessing.OneHotTransformer().transform
                  ]

        for check in checks:
            with pytest.raises(AssertionError):
                check(inp)


class TestOneHotEncodingUnAmbig(object):

    spacer_alpha = IUPAC.unambiguous_rna
    target_alpha = IUPAC.unambiguous_dna
    processor = partial(preprocessing.one_hot_encode_row)
    cor_shape = 4 * 4 * 20 + 4*4

    def check_shape(self, vals):
        assert vals.shape[0] == self.cor_shape

    def get_pam_pos(self, pam):

        sz = len(self.spacer_alpha.letters) * len(self.target_alpha.letters)
        pam_order = list(product(sorted(self.target_alpha.letters), repeat=2))
        pos = next(num for num, _pam in enumerate(pam_order) if (''.join(_pam)) == pam[-2:])
        return 20*sz + pos

    def test_encoding(self):

        spacer = Seq('A' * 20, alphabet=generic_rna)
        target = Seq('A' * 20 + 'AGG', alphabet=generic_dna)

        cor = np.zeros(self.cor_shape)
        sz = len(self.spacer_alpha.letters) * len(self.target_alpha.letters)

        locs = np.arange(0, 20 * sz, sz)
        cor[locs] = True

        cor[self.get_pam_pos('AGG')] = True  # GG

        res = self.processor(spacer, target)
        self.check_shape(res)

        np.testing.assert_array_equal(cor.astype(bool), res)

    def test_more_encoding(self):

        spacer = Seq('U' + 'A' * 19, alphabet=generic_rna)
        target = Seq('A' * 20 + 'AGG', alphabet=generic_dna)

        cor = np.zeros(self.cor_shape)
        sz = len(self.spacer_alpha.letters) * len(self.target_alpha.letters)
        locs = np.arange(0, 20 * sz, sz)
        cor[locs] = True
        cor[0] = False

        ua_pos = 3*len(self.target_alpha.letters)
        cor[ua_pos] = True

        cor[self.get_pam_pos('AGG')] = True  # GG

        res = self.processor(spacer, target)
        self.check_shape(res)

        np.testing.assert_array_equal(cor.astype(bool), res)

    def test_PAM_encoding(self):

        spacer = Seq('U' + 'A' * 19, alphabet=generic_rna)

        sz = len(self.spacer_alpha.letters) * len(self.target_alpha.letters)
        locs = np.arange(0, 20 * sz, sz)
        ua_pos = 3*len(self.target_alpha.letters)

        pams = product(sorted(self.target_alpha.letters), repeat=2)
        for pos, (p1, p2) in enumerate(pams):
            cor = np.zeros(self.cor_shape)
            cor[locs[1:]] = True
            cor[ua_pos] = True

            cor[20*sz + pos] = True

            target = Seq('A' * 20 + 'A' + p1 + p2, alphabet=generic_dna)

            res = self.processor(spacer, target)
            self.check_shape(res)

            np.testing.assert_array_equal(cor.astype(bool), res)

    def test_ambigious_target(self):

        spacer = Seq('A' * 20, alphabet=generic_rna)
        target = Seq('N' + 'A' * 19 + 'AGG', alphabet=generic_dna)

        with pytest.raises(AssertionError):
            preprocessing.one_hot_encode_row(spacer, target)


class TestOneHotEncodingAmbig(TestOneHotEncodingUnAmbig):

    spacer_alpha = IUPAC.unambiguous_rna
    target_alpha = IUPAC.ambiguous_dna
    cor_shape = 4 * 15 * 20 + 15*15

    processor = partial(preprocessing.one_hot_encode_row,
                        spacer_alphabet=IUPAC.unambiguous_rna,
                        target_alphabet=IUPAC.ambiguous_dna)


    def test_ambigious_target(self):

        spacer = Seq('A' * 20, alphabet=generic_rna)
        target = Seq('N' + 'A' * 19 + 'AGG', alphabet=generic_dna)

        res = self.processor(spacer, target)
        self.check_shape(res)

        cor = np.zeros(self.cor_shape)
        sz = len(self.spacer_alpha.letters) * len(self.target_alpha.letters)
        locs = np.arange(0, 20 * sz, sz)
        cor[locs] = True
        cor[0] = False
        cor[8] = True #AN
        cor[self.get_pam_pos('AGG')] = True  # GG

        np.testing.assert_array_equal(cor.astype(bool), res)

    def test_ambigious_pam(self):

        spacer = Seq('A' * 20, alphabet=generic_rna)
        target = Seq('N' + 'A' * 19 + 'ARG', alphabet=generic_dna)

        res = self.processor(spacer, target)
        self.check_shape(res)

        cor = np.zeros(self.cor_shape)
        sz = len(self.spacer_alpha.letters) * len(self.target_alpha.letters)
        locs = np.arange(0, 20 * sz, sz)
        cor[locs] = True
        cor[0] = False
        cor[8] = True #AN
        cor[self.get_pam_pos('ARG')] = True  # GG

        np.testing.assert_array_equal(cor.astype(bool), res)


class TestOneHotTransformer(object):

    def test_transforming(self):

        spacer = Seq('U' + 'A' * 19, alphabet=generic_rna)
        target = Seq('A' * 20 + 'AGG', alphabet=generic_dna)

        cor = np.zeros(21 * 16)
        locs = np.arange(0, 20 * 16, 16)
        cor[locs] = True
        cor[0] = False
        cor[12] = True
        cor[-6] = True  # GG

        inp = np.array([[spacer, target],
                        [spacer, target],
                        [spacer, target]])

        hot_encoder = preprocessing.OneHotTransformer()
        res = hot_encoder.transform(inp)

        assert res.shape == (3, 21 * 16)

        for row in range(3):
            np.testing.assert_array_equal(cor.astype(bool), res[row, :])


class TestMatchingEncoding(object):
    def test_encoding(self):
        spacer = Seq('A' * 20, alphabet=generic_rna)
        target = Seq('A' * 20 + 'AGG', alphabet=generic_dna)
        cor = np.array([True] * 21)

        res = preprocessing.match_encode_row(spacer, target)
        assert res.shape == (20 + 1,)

        np.testing.assert_array_equal(cor.astype(bool), res)

    def test_more_encoding(self):
        spacer = Seq('U' + 'A' * 19, alphabet=generic_rna)
        target = Seq('A' * 20 + 'AGG', alphabet=generic_dna)

        cor = np.array([False] + [True] * 20)

        res = preprocessing.match_encode_row(spacer, target)

        assert res.shape == (20 + 1,)

        np.testing.assert_array_equal(cor.astype(bool), res)

    def test_U_encoding(self):
        spacer = Seq('U' + 'A' * 19, alphabet=generic_rna)
        target = Seq('T' + 'A' * 19 + 'AGG', alphabet=generic_dna)

        cor = np.array([True] * 21)

        res = preprocessing.match_encode_row(spacer, target)

        assert res.shape == (20 + 1,)

        np.testing.assert_array_equal(cor.astype(bool), res)

    def test_transforming(self):
        spacer = Seq('U' + 'A' * 19, alphabet=generic_rna)
        target = Seq('A' * 20 + 'AGG', alphabet=generic_dna)

        cor = np.array([False] + [True] * 20)

        inp = np.array([[spacer, target],
                        [spacer, target],
                        [spacer, target]])
        hot_encoder = preprocessing.MatchingTransformer()
        res = hot_encoder.transform(inp)

        assert res.shape == (3, 20 + 1)

        for row in range(3):
            np.testing.assert_array_equal(cor.astype(bool), res[row, :])


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
    return ''.join(np.random.choice(list('ACGT'), size=bp))


class TestLocate(object):
    def make_basic(self):
        # prevent Heisenbugs
        np.random.seed(0)

        locus = [make_random_seq(50) + 'TTTT' + 'A' * 20 + 'CGG' + 'TTTT' + make_random_seq(50),
                 make_random_seq(12) + 'TTTT' + 'C' * 19 + 'T' + 'CGG' + 'TTTT' + make_random_seq(50),
                 make_random_seq(75),
                 make_random_seq(25) + 'TTTT' + 'T' + 'A' * 19 + 'TGG' + 'TTTT' + make_random_seq(50)]
        locus = [SeqRecord(Seq(s, alphabet=generic_dna), id=str(n)) for n, s in enumerate(locus)]

        spacers = [Seq('A' * 20, alphabet=generic_rna),
                   Seq('C' * 19 + 'U', alphabet=generic_rna),
                   Seq('C' * 19 + 'U', alphabet=generic_rna),
                   Seq('A' * 20, alphabet=generic_rna)]

        cor_pos = [54, 16, np.nan, 29]
        cor_strand = [1, 1, np.nan, 1]

        cor_target = [Seq('A' * 20 + 'CGG', alphabet=generic_dna),
                      Seq('C' * 19 + 'T' + 'CGG', alphabet=generic_dna),
                      np.nan,
                      Seq('T' + 'A' * 19 + 'TGG', alphabet=generic_dna)]

        cor_spacer = [Seq('A' * 20, alphabet=generic_rna),
                      Seq('C' * 19 + 'U', alphabet=generic_rna),
                      np.nan,
                      Seq('A' * 20, alphabet=generic_rna)]

        return spacers, locus, cor_target, cor_spacer, cor_pos, cor_strand

    @pytest.mark.skipif(utils._missing_casoffinder(), reason="Need CasOff installed")
    def test_basic(self):
        spacers, locus, cor_target, cor_spacer, cor_pos, cor_strand = self.make_basic()

        X = np.array(list(zip(spacers, locus)))
        estimator = CFDEstimator.build_pipeline()

        nX, loc, _ = preprocessing.locate_hits_in_array(X, estimator, mismatches=6)

        assert nX.shape == (4, 2)
        assert loc.shape == (4, 2)

        np.testing.assert_array_equal(cor_pos, loc[:, 0])
        np.testing.assert_array_equal(cor_strand, loc[:, 1])

        cX = pd.DataFrame(list(zip(cor_spacer, cor_target))).values

        mask = np.array([True, True, False, True])
        np.testing.assert_array_equal(cX[mask, :], nX[mask, :])

        assert np.isnan(cX[2, 0])
        assert np.isnan(cX[2, 1])

    def test_exhaustive(self):
        spacers, locus, cor_target, cor_spacer, cor_pos, cor_strand = self.make_basic()

        X = np.array(list(zip(spacers, locus)))
        estimator = CFDEstimator.build_pipeline()

        nX, loc, _ = preprocessing.locate_hits_in_array(X, estimator, exhaustive=True)

        assert nX.shape == (4, 2)
        assert loc.shape == (4, 2)

        mask = np.array([0, 1, 3])

        np.testing.assert_array_equal(np.array(cor_pos)[mask], loc[mask, 0])
        np.testing.assert_array_equal(np.array(cor_strand)[mask], loc[mask, 1])

        cX = pd.DataFrame(list(zip(cor_spacer, cor_target))).values

        np.testing.assert_array_equal(cX[mask, :], nX[mask, :])
