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

        seqdf = pd.DataFrame([(Seq('A'*100), 10, 110),
                              (Seq('T'*100), 11, 111)],
                              columns = ['Seq', 'Start', 'Stop'])

        grnadf = pd.DataFrame([(Seq('A'*20), 30, 52),
                              (Seq('T'*20), 30, 52)],
                              columns = ['gRNA', 'Start', 'Stop'])
        est = estimators.CFDEstimator.build_pipeline()

        return seqdf, grnadf, est


    def test_column_names(self):

        seqdf, grnadf, est = self.make_basic_dfs()

        evaluators.positional_aggregation(seqdf, grnadf, est)

        with pytest.raises(AssertionError):
            bseqdf = pd.DataFrame([(Seq('A'*30), 10, 40),
                              (Seq('T'*30), 11, 41)],
                              columns = ['Sequence', 'Start', 'Stop'])
            evaluators.positional_aggregation(bseqdf, grnadf, est)

        with pytest.raises(AssertionError):
            bgrnadf = pd.DataFrame([(Seq('A'*20), 10, 15),
                                   (Seq('T'*20), 11, 17)],
                                  columns = ['gRNAs', 'Start', 'Stop'])
            evaluators.positional_aggregation(seqdf, bgrnadf, est)


    def test_seq_column_type(self):

        seqdf, grnadf, est = self.make_basic_dfs()

        with pytest.raises(AssertionError):
            bseqdf = pd.DataFrame([('A'*30, 10, 40),
                              ('T'*30, 11, 41)],
                              columns = ['Seq', 'Start', 'Stop'])
            evaluators.positional_aggregation(bseqdf, grnadf, est)

        with pytest.raises(AssertionError):
            bgrnadf = pd.DataFrame([('A'*20, 10, 15),
                                    ('T'*20, 11, 17)],
                                   columns = ['gRNA', 'Start', 'Stop'])
            evaluators.positional_aggregation(seqdf, bgrnadf, est)

    def test_iterate_overlaps(self):

        seqdf = pd.DataFrame([('S1', 50, 200),
                              ('S2', 150, 250),
                              ('S3', 350, 1100),
                              ('S4', 500, 700),
                              ('S5', 400, 900)],
                             columns = ['Seq', 'Start', 'Stop'])

        grnadf = pd.DataFrame([('g1', 175, 195),
                               ('g2', 600, 620)],
                              columns = ['gRNA', 'Start', 'Stop'])

        cors = [('g1',  {'S1', 'S2'}),
                ('g2',  {'S3', 'S4', 'S5'})]

        found = list(evaluators._iterate_grna_seq_overlaps(seqdf, grnadf, 20))
        assert len(found) == 2

        for (grow, sdf), (gkey, cor_seqs) in zip(found, cors):
            assert grow['gRNA'] == gkey
            assert set(sdf['Seq']) == cor_seqs

    def make_complicated_dfs(self):

        grnaA = 'A'*10 + 'C'*10
        grnaB = 'T'*10 + 'A'*10

        big_seq = 'G'*100 + grnaA + 'G'*320 + grnaB + 'G'*100

        regions = [(50, 250),
                   (110, 300),
                   (200, 350),
                   (300, 500)]
        seqs = []
        correct_calls = []
        for num, (start, stop) in enumerate(regions):
            seqs.append({'Start': start, 'Stop': stop, 'Seq': Seq(big_seq[start:stop]),
                         'Num': num})
            for grna in [grnaA, grnaB]:
                pos = seqs[-1]['Seq'].find(grna)
                correct_calls.append({'Num': num, 'gRNA': grna,
                                      'CorPosition': pos if pos >=0 else np.nan,
                                      'IsPresent': pos != -1})

        wrong_seq = 'G'*len(big_seq)
        for num, (start, stop) in enumerate(regions, num):
            seqs.append({'Start': start, 'Stop': stop, 'Seq': Seq(wrong_seq[start:stop]),
                         'Num': num})
            for grna in [grnaA, grnaB]:
                correct_calls.append({'Num': num, 'gRNA': grna,
                                      'CorPosition': np.nan,
                                      'IsPresent': False})

        seqdf = pd.DataFrame(seqs)
        grnadf = pd.DataFrame([{'gRNA': Seq(grnaA), 'Start': 100, 'Stop': 120},
                               {'gRNA': Seq(grnaB), 'Start': 450, 'Stop': 470}])
        cordf = pd.DataFrame(correct_calls)

        return seqdf, grnadf, cordf

    def test_positional_aggregation(self):

        seqdf, grnadf, cordf = self.make_complicated_dfs()
        est = estimators.MismatchEstimator.build_pipeline(miss_non_seed=0)

        results = evaluators.positional_aggregation(seqdf, grnadf, est, overlap=-1)
        merged = pd.merge(results, cordf, on = ['Num', 'gRNA'], how = 'outer')

        assert merged.loc[merged['IsPresent'], 'Value'].all()
