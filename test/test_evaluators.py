from crisprtree import preprocessing
from crisprtree import estimators
from crisprtree import evaluators
from sklearn.pipeline import Pipeline
from Bio.Seq import reverse_complement, Seq
import numpy as np
import pandas as pd
from pandas.util.testing import assert_series_equal, assert_index_equal
import pytest


def build_estimator():

    mod = Pipeline(steps = [('transform', preprocessing.MatchingTransformer()),
                            ('predict', estimators.MismatchEstimator())])
    return mod


class TestCheckgRNA(object):


    def test_basic(self):

        gRNA = 'A'*20

        seqs = ['T'*10 + gRNA + 'TGG' + 'T'*5,
                'T'*15 + gRNA + 'TGG' + 'T'*5,
                'C'*30
                ]
        corr = pd.Series([True, True, False])

        est = build_estimator()

        res = evaluators.check_grna_across_seqs(gRNA, seqs, est)
        assert list(res.columns) == ['Value', 'Target', 'Position', 'Strand']

        assert_series_equal(corr, res['Value'], check_names=False)

        assert_series_equal(pd.Series([10, 15]), res['Position'].iloc[:2],
                            check_names=False)

        assert_series_equal(pd.Series(['+', '+']), res['Strand'].iloc[:2],
                            check_names=False)

    def test_accepts_grna_seq(self):
        gRNA = 'A'*20

        seqs = ['T'*10 + gRNA + 'TGG' + 'T'*5,
                'T'*15 + gRNA + 'TGG' + 'T'*5,
                'C'*30
                ]
        corr = pd.Series([True, True, False])

        est = build_estimator()

        res = evaluators.check_grna_across_seqs(Seq(gRNA), seqs, est)
        assert list(res.columns) == ['Value', 'Target', 'Position', 'Strand']

        assert_series_equal(corr, res['Value'], check_names=False)

        assert_series_equal(pd.Series([10, 15]), res['Position'].iloc[:2],
                            check_names=False)

        assert_series_equal(pd.Series(['+', '+']), res['Strand'].iloc[:2],
                            check_names=False)


    def test_basic_RC(self):

        gRNA = 'A'*20

        seqs = ['T'*10 + gRNA + 'TGG' + 'T'*5,
                'T'*15 + gRNA + 'TGG' + 'T'*5,
                'C'*30
                ]
        seqs += [reverse_complement(seq) for seq in seqs]

        corr = pd.Series([True, True, False]*2)

        est = build_estimator()

        res = evaluators.check_grna_across_seqs(gRNA, seqs, est)

        assert_series_equal(corr, res['Value'], check_names=False)

    def test_accepts_short_seqs(self):

        gRNA = 'A'*20

        seqs = ['T'*10 + gRNA + 'TGG' + 'T'*5,
                'T'*15 + gRNA + 'TGG' + 'T'*5,
                'C'*30,
                'G'*12
                ]
        seqs += [reverse_complement(seq) for seq in seqs]

        corr = pd.Series([True, True, False, False]*2)

        est = build_estimator()

        res = evaluators.check_grna_across_seqs(gRNA, seqs, est)

        assert_series_equal(corr, res['Value'], check_names=False)

    def test_carries_series_index(self):

        gRNA = 'A'*20

        seqs = ['T'*10 + gRNA + 'TGG' + 'T'*5,
                'T'*15 + gRNA + 'TGG' + 'T'*5,
                'C'*30,
                'G'*12
                ]
        seqs += [reverse_complement(seq) for seq in seqs]

        index = pd.Index(['Seq%i' % i for i in range(len(seqs))], name='SeqIndex')
        seqs = pd.Series(seqs, index=index)

        corr = pd.Series([True, True, False, False]*2,
                         index=index)

        est = build_estimator()

        res = evaluators.check_grna_across_seqs(gRNA, seqs, est)

        assert_series_equal(corr, res['Value'], check_names=False)
        assert_index_equal(corr.index, res.index)

    def test_accepts_index(self):

        gRNA = 'A'*20

        seqs = ['T'*10 + gRNA + 'TGG' + 'T'*5,
                'T'*15 + gRNA + 'TGG' + 'T'*5,
                'C'*30,
                'G'*12
                ]
        seqs += [reverse_complement(seq) for seq in seqs]

        index = pd.Index(['Seq%i' % i for i in range(len(seqs))], name='SeqIndex')

        corr = pd.Series([True, True, False, False]*2,
                         index=index)

        est = build_estimator()

        res = evaluators.check_grna_across_seqs(gRNA, seqs, est, index=index)

        assert_series_equal(corr, res['Value'], check_names=False)
        assert_index_equal(corr.index, res.index)

    def test_accepts_list_index(self):

        gRNA = 'A'*20

        seqs = ['T'*10 + gRNA + 'TGG' + 'T'*5,
                'T'*15 + gRNA + 'TGG' + 'T'*5,
                'C'*30,
                'G'*12
                ]
        seqs += [reverse_complement(seq) for seq in seqs]

        index = ['Seq%i' % i for i in range(len(seqs))]

        corr = pd.Series([True, True, False, False]*2,
                         index=index)

        est = build_estimator()

        res = evaluators.check_grna_across_seqs(gRNA, seqs, est, index=index)

        assert_series_equal(corr, res['Value'], check_names=False)
        assert_index_equal(corr.index, res.index)


class TestPositionalAgg(object):

    def make_basic_dfs(self):

        seqdf = pd.DataFrame([(Seq('A'*30), 10, 40),
                              (Seq('T'*30), 11, 41)],
                              columns = ['Seq', 'Start', 'Stop'])

        grnadf = pd.DataFrame([(Seq('A'*20), 10, 15),
                              (Seq('T'*20), 11, 17)],
                              columns = ['gRNA', 'Start', 'Stop'])
        return seqdf, grnadf


    def test_column_names(self):

        seqdf, grnadf = self.make_basic_dfs()

        evaluators.positional_aggregation(seqdf, grnadf)

        with pytest.raises(AssertionError):
            bseqdf = pd.DataFrame([(Seq('A'*30), 10, 40),
                              (Seq('T'*30), 11, 41)],
                              columns = ['Sequence', 'Start', 'Stop'])
            evaluators.positional_aggregation(bseqdf, grnadf)

        with pytest.raises(AssertionError):
            bgrnadf = pd.DataFrame([(Seq('A'*20), 10, 15),
                                   (Seq('T'*20), 11, 17)],
                                  columns = ['gRNAs', 'Start', 'Stop'])
            evaluators.positional_aggregation(seqdf, bgrnadf)


    def test_seq_column_type(self):

        seqdf, grnadf = self.make_basic_dfs()

        with pytest.raises(AssertionError):
            bseqdf = pd.DataFrame([('A'*30, 10, 40),
                              ('T'*30, 11, 41)],
                              columns = ['Seq', 'Start', 'Stop'])
            evaluators.positional_aggregation(bseqdf, grnadf)

        with pytest.raises(AssertionError):
            bgrnadf = pd.DataFrame([('A'*20, 10, 15),
                                    ('T'*20, 11, 17)],
                                   columns = ['gRNA', 'Start', 'Stop'])
            evaluators.positional_aggregation(seqdf, bgrnadf)





