from __future__ import division

import os
from itertools import cycle
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
import yaml
from Bio import Alphabet
from Bio.Seq import Seq
from sklearn.pipeline import Pipeline

from crisprtree import preprocessing
from crisprtree import estimators


def make_match_array_from_seqs(spacer, seqs):
    """
    Utility function for creating a MatchArray (Nx21 boolean) from a list
    of sequences.

    Parameters
    ----------
    spacer : Seq
        A 20 bp gRNA
    seqs : iter
        An iterable of 23bp target sequences

    Returns
    -------
    np.array

    """

    seq_array = np.array(list(zip(cycle([spacer]), seqs)))
    return preprocessing.MatchingTransformer().transform(seq_array)


def make_onehot_array_from_seqs(spacer, seqs):
    """
    Utility function for creating a OneHotArray (Nx336 boolean) from a list
    of sequences.

    Parameters
    ----------
    spacer : str
        A 20 bp gRNA
    seqs : iter
        An iterable of 23bp target sequences

    Returns
    -------
    np.array

    """

    seq_array = np.array(list(zip(cycle([spacer]), seqs)))
    return preprocessing.OneHotTransformer().transform(seq_array)


class BaseChecker(object):
    hits = [Seq('A' * 20 + 'AGG', alphabet=Alphabet.generic_dna),  # Perfect hit
            Seq('A' * 19 + 'T' + 'AGG', alphabet=Alphabet.generic_dna),  # One miss in seed
            Seq('T' + 'A' * 19 + 'AGG', alphabet=Alphabet.generic_dna),  # One miss outside seed
            Seq('TTT' + 'A' * 17 + 'AGG', alphabet=Alphabet.generic_dna),  # Three miss outside seed
            Seq('A' * 20 + 'ATG', alphabet=Alphabet.generic_dna)  # No PAM
            ]
    expected = [0, 0, 0, 0, 0]
    prob = None

    estimator = None
    pipeline = None

    def _make_match_array(self, spacer, hits):
        return make_match_array_from_seqs(spacer, hits)

    def test_raises_value_error_on_wrong_size(self):

        check = np.ones((5, 20))

        with pytest.raises(ValueError):
            self.estimator.predict(check)

    def test_basic_predict(self):
        spacer = Seq('A' * 20, alphabet=Alphabet.generic_rna)

        if self.estimator is None:
            raise NotImplementedError

        match_array = self._make_match_array(spacer, self.hits)

        res = self.estimator.predict(match_array)
        np.testing.assert_array_equal(res, self.expected)

    def test_predict_proba(self):

        if self.prob is not None:
            spacer = Seq('A' * 20, alphabet=Alphabet.generic_rna)

            if self.estimator is None:
                raise NotImplementedError

            match_array = self._make_match_array(spacer, self.hits)

            res = self.estimator.predict_proba(match_array)
            np.testing.assert_almost_equal(res, self.prob, decimal=3)

    def test_change_cutoff(self):

        if self.prob is not None:
            spacer = Seq('A' * 20, alphabet=Alphabet.generic_rna)

            if self.estimator is None:
                raise NotImplementedError

            match_array = self._make_match_array(spacer, self.hits)

            self.estimator.cutoff = 0.2
            res = self.estimator.predict(match_array)
            np.testing.assert_array_equal(res, np.array(self.prob) > 0.2)

            self.estimator.cutoff = 0.5
            res = self.estimator.predict(match_array)
            np.testing.assert_array_equal(res, np.array(self.prob) > 0.5)

            self.estimator.cutoff = 0.8
            res = self.estimator.predict(match_array)
            np.testing.assert_array_equal(res, np.array(self.prob) > 0.8)

    def test_build_pipeline(self):
        spacer = Seq('A' * 20, alphabet=Alphabet.generic_rna)

        if self.pipeline is None:
            raise NotImplementedError

        seq_array = np.array(list(zip(cycle([spacer]), self.hits)))

        res = self.pipeline.predict(seq_array)
        np.testing.assert_array_equal(res, self.expected)


class TestMismatchEstimator(BaseChecker):
    expected = [True, False, True, False, False]
    prob = None

    estimator = estimators.MismatchEstimator(seed_len=4, miss_seed=0,
                                             miss_tail=2, pam='NGG')
    pipeline = estimators.MismatchEstimator.build_pipeline(seed_len=4,
                                                           miss_seed=0,
                                                           miss_tail=2,
                                                           pam='NGG')

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
        spacer = Seq('A' * 20, alphabet=Alphabet.generic_rna)
        hits = [Seq('A' * 20 + 'AGG', alphabet=Alphabet.generic_dna),
                Seq('A' * 19 + 'T' + 'AGG', alphabet=Alphabet.generic_dna),
                Seq('A' * 18 + 'TA' + 'AGG', alphabet=Alphabet.generic_dna),
                Seq('A' * 17 + 'TAA' + 'AGG', alphabet=Alphabet.generic_dna),
                Seq('A' * 16 + 'TAAA' + 'AGG', alphabet=Alphabet.generic_dna)]

        match_array = self._make_match_array(spacer, hits)

        mod = estimators.MismatchEstimator(seed_len=2,
                                           miss_seed=0,
                                           miss_tail=2,
                                           pam='NGG')
        res = mod.predict(match_array)
        expected = [True, False, False, True, True]
        np.testing.assert_array_equal(res, expected)

        mod = estimators.MismatchEstimator(seed_len=3,
                                           miss_seed=0,
                                           miss_tail=2,
                                           pam='NGG')
        res = mod.predict(match_array)
        expected = [True, False, False, False, True]
        np.testing.assert_array_equal(res, expected)

    def test_set_seed_mismatch(self):
        spacer = Seq('A' * 20, alphabet=Alphabet.generic_rna)
        hits = [Seq('A' * 20 + 'AGG', alphabet=Alphabet.generic_dna),
                Seq('A' * 19 + 'T' + 'AGG', alphabet=Alphabet.generic_dna),
                Seq('A' * 18 + 'TT' + 'AGG', alphabet=Alphabet.generic_dna),
                Seq('A' * 17 + 'TTT' + 'AGG', alphabet=Alphabet.generic_dna)]

        match_array = self._make_match_array(spacer, hits)

        mod = estimators.MismatchEstimator(seed_len=4,
                                           miss_seed=1,
                                           miss_tail=2,
                                           pam='NGG')
        res = mod.predict(match_array)
        expected = [True, True, False, False]
        np.testing.assert_array_equal(res, expected)

        mod = estimators.MismatchEstimator(seed_len=4,
                                           miss_seed=2,
                                           miss_tail=2,
                                           pam='NGG')
        res = mod.predict(match_array)
        expected = [True, True, True, False]
        np.testing.assert_array_equal(res, expected)

    def test_set_non_seed_mismatch(self):
        spacer = Seq('A' * 20, alphabet=Alphabet.generic_rna)
        hits = [Seq('A' * 20 + 'AGG', alphabet=Alphabet.generic_dna),
                Seq('T' + 'A' * 19 + 'AGG', alphabet=Alphabet.generic_dna),
                Seq('TT' + 'A' * 18 + 'AGG', alphabet=Alphabet.generic_dna),
                Seq('TTT' + 'A' * 17 + 'AGG', alphabet=Alphabet.generic_dna)]

        match_array = make_match_array_from_seqs(spacer, hits)

        mod = estimators.MismatchEstimator(seed_len=4,
                                           miss_seed=0,
                                           miss_tail=2,
                                           pam='NGG')
        res = mod.predict(match_array)
        expected = [True, True, True, False]
        np.testing.assert_array_equal(res, expected)

        mod = estimators.MismatchEstimator(seed_len=4,
                                           miss_seed=0,
                                           miss_tail=1,
                                           pam='NGG')
        res = mod.predict(match_array)
        expected = [True, True, False, False]
        np.testing.assert_array_equal(res, expected)


class TestMITestimator(BaseChecker):
    hits = [Seq('A' * 20 + 'AGG', alphabet=Alphabet.generic_dna),
            Seq('A' * 19 + 'T' + 'CGG', alphabet=Alphabet.generic_dna),
            Seq('T' + 'A' * 19 + 'GGG', alphabet=Alphabet.generic_dna),
            Seq('TT' + 'A' * 18 + 'GGG', alphabet=Alphabet.generic_dna),
            Seq('A' * 5 + 'TT' + 'A' * 13 + 'GGG', alphabet=Alphabet.generic_dna)]
    expected = [True, False, True, True, False]
    prob = [1.0, 0.417, 1, 1, 0.413]
    estimator = estimators.MITEstimator()
    pipeline = estimators.MITEstimator.build_pipeline()

    def test_requires_pam(self):
        spacer = Seq('T' + 'A' * 19, alphabet=Alphabet.generic_rna)
        hits = [Seq('A' * 20 + 'AGG', alphabet=Alphabet.generic_dna),
                Seq('A' * 19 + 'T' + 'CGG', alphabet=Alphabet.generic_dna),
                Seq('T' + 'A' * 19 + 'GGG', alphabet=Alphabet.generic_dna),
                Seq('A' * 20 + 'ATG', alphabet=Alphabet.generic_dna), ]

        match_array = make_match_array_from_seqs(spacer, hits)

        mit_est = estimators.MITEstimator(cutoff=0.05)
        mit_cut = mit_est.predict(match_array)

        cor_prob = [True, True, True, False]

        np.testing.assert_equal(cor_prob, mit_cut)


class TestKineticEstimator(BaseChecker):
    hits = [Seq('A' * 20 + 'AGG', alphabet=Alphabet.generic_dna),
            Seq('A' * 19 + 'T' + 'CGG', alphabet=Alphabet.generic_dna),
            Seq('T' + 'A' * 19 + 'GGG', alphabet=Alphabet.generic_dna),
            Seq('TT' + 'A' * 18 + 'GGG', alphabet=Alphabet.generic_dna),
            Seq('A' * 5 + 'TT' + 'A' * 13 + 'GGG', alphabet=Alphabet.generic_dna)]
    expected = [True, False, True, True, True]
    prob = [1.0, 1.12701e-8, 0.7399, 0.5475, 0.5447311]

    estimator = estimators.KineticEstimator()
    pipeline = estimators.KineticEstimator.build_pipeline()

    def test_requires_pam(self):
        spacer = Seq('A' * 20, alphabet=Alphabet.generic_rna)
        hits = [Seq('A' * 20 + 'AGG', alphabet=Alphabet.generic_dna),
                Seq('A' * 19 + 'T' + 'CGG', alphabet=Alphabet.generic_dna),
                Seq('T' + 'A' * 19 + 'GGG', alphabet=Alphabet.generic_dna),
                Seq('A' * 20 + 'ATG', alphabet=Alphabet.generic_dna), ]

        match_array = make_match_array_from_seqs(spacer, hits)

        est = estimators.KineticEstimator(cutoff=0.50)
        cut = est.predict(match_array)

        cor_prob = [True, False, True, False]

        np.testing.assert_equal(cor_prob, cut)


class TestCFDEstimator(BaseChecker):
    hits = [Seq('A' * 20 + 'AGG', alphabet=Alphabet.generic_dna),
            Seq('A' * 19 + 'T' + 'CGG', alphabet=Alphabet.generic_dna),
            Seq('T' + 'A' * 19 + 'GGG', alphabet=Alphabet.generic_dna),
            Seq('TT' + 'A' * 18 + 'GGG', alphabet=Alphabet.generic_dna),
            Seq('A' * 5 + 'TT' + 'A' * 13 + 'GGG', alphabet=Alphabet.generic_dna),
            Seq('A' * 20 + 'GAG', alphabet=Alphabet.generic_dna)]
    expected = [True, False, True, False, False, False]
    prob = [1.0, 0.6, 1.0, 0.727, 0.714 * 0.4375, 0.259]

    estimator = estimators.CFDEstimator()
    pipeline = estimators.CFDEstimator.build_pipeline()

    def _make_match_array(self, spacer, hits):
        return make_onehot_array_from_seqs(spacer, hits)

    def test_loading(self):
        mod = estimators.CFDEstimator()
        assert mod.score_vector.shape == (336,)
        np.testing.assert_approx_equal(mod.score_vector.sum(), 217.9692)
