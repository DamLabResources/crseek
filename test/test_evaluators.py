from crisprtree import preprocessing
from crisprtree import estimators
from crisprtree import evaluators
from test.test_preprocessing import make_random_seq
from sklearn.pipeline import Pipeline
from Bio.Seq import reverse_complement, Seq
from Bio.SeqRecord import SeqRecord
from Bio import Alphabet
import numpy as np
import pandas as pd
from pandas.util.testing import assert_series_equal, assert_index_equal
import pytest
from copy import deepcopy


def build_estimator():

    mod = Pipeline(steps = [('transform', preprocessing.MatchingTransformer()),
                            ('predict', estimators.MismatchEstimator())])
    return mod


def do_rev_comp(seqs):

    ndata = []
    for seqR in seqs:
        ndata.append(deepcopy(seqR))
        ndata[-1].seq = reverse_complement(ndata[-1].seq)
        ndata[-1].id = ndata[-1].id + '-R'

    return ndata

class TestCheckgRNA(object):

    def get_basic_info(self):

        spacer = Seq('A'*20, alphabet = Alphabet.generic_rna)

        loci = [SeqRecord(Seq('T'*10 + str(spacer) + 'TGG' + 'T'*5,
                              alphabet = Alphabet.generic_dna),
                          id = 'Seq1'),
                SeqRecord(Seq('T'*15 + str(spacer) + 'TGG' + 'T'*5,
                              alphabet = Alphabet.generic_dna),
                          id = 'Seq2'),
                SeqRecord(Seq('C'*30,
                              alphabet = Alphabet.generic_dna),
                          id = 'Seq3')
                ]
        corr = pd.Series([True, True, np.nan],
                         index = [s.id + ' ' + s.description for s in loci])
        return spacer, loci, corr

    def test_basic(self):

        spacer, loci, corr = self.get_basic_info()

        est = build_estimator()

        res = evaluators.check_spacer_across_loci(spacer, loci, est)
        assert list(res.columns) == ['left', 'strand', 'spacer', 'target', 'score']

        assert_series_equal(corr, res['score'], check_names=False)

        assert_series_equal(pd.Series([10, 15], index = corr.index[:2]),
                            res['left'].iloc[:2],
                            check_dtype = False,
                            check_names=False)

        assert_series_equal(pd.Series([1, 1], index = corr.index[:2]),
                            res['strand'].iloc[:2],
                            check_dtype = False,
                            check_names=False)

    def test_basic_RC(self):

        spacer, loci, _ = self.get_basic_info()
        loci += do_rev_comp(loci)

        corr = pd.Series([True, True, np.nan]*2,
                         [s.id + ' ' + s.description for s in loci])

        est = build_estimator()

        res = evaluators.check_spacer_across_loci(spacer, loci, est)

        assert_series_equal(corr, res['score'],
                            check_names=False,
                            check_dtype = False)

    def test_accepts_short_seqs(self):

        spacer, loci, _ = self.get_basic_info()

        loci.append(SeqRecord(Seq('G'*12,
                                  alphabet = Alphabet.generic_dna),
                              id = 'Seq4'))
        loci += do_rev_comp(loci)

        corr = pd.Series([True, True, np.nan, np.nan]*2,
                         index = [s.id + ' ' + s.description for s in loci])

        est = build_estimator()
        res = evaluators.check_spacer_across_loci(spacer, loci, est)

        assert_series_equal(corr, res['score'],
                            check_dtype = False,
                            check_names=False)

    def test_carries_series_index(self):

        spacer, loci, _ = self.get_basic_info()

        loci.append(SeqRecord(Seq('G'*12,
                                  alphabet = Alphabet.generic_dna),
                              id = 'Seq4'))
        loci += do_rev_comp(loci)

        index = pd.Index(['Seq%i' % i for i in range(len(loci))], name='SeqIndex')
        loci = pd.Series(loci, index=index)

        corr = pd.Series([True, True, np.nan, np.nan]*2,
                         index=index)

        est = build_estimator()

        res = evaluators.check_spacer_across_loci(spacer, loci, est)

        assert_series_equal(corr, res['score'],
                            check_names=False,
                            check_dtype = False)
        assert_index_equal(corr.index, res.index)

    def test_accepts_index(self):

        spacer, loci, _ = self.get_basic_info()

        loci.append(SeqRecord(Seq('G'*12,
                                  alphabet = Alphabet.generic_dna),
                              id = 'Seq4'))
        loci += do_rev_comp(loci)

        index = pd.Index(['Seq%i' % i for i in range(len(loci))], name='SeqIndex')
        loci = pd.Series(loci, index=index)

        corr = pd.Series([True, True, np.nan, np.nan]*2,
                         index=index)

        est = build_estimator()

        res = evaluators.check_spacer_across_loci(spacer, loci, est, index = index)

        assert_series_equal(corr, res['score'],
                            check_dtype = False,
                            check_names=False)
        assert_index_equal(corr.index, res.index)

    def test_accepts_list_index(self):

        spacer, loci, _ = self.get_basic_info()

        loci.append(SeqRecord(Seq('G'*12,
                                  alphabet = Alphabet.generic_dna),
                              id = 'Seq4'))
        loci += do_rev_comp(loci)

        index = ['Seq%i' % i for i in range(len(loci))]
        loci = pd.Series(loci, index=index)

        corr = pd.Series([True, True, np.nan, np.nan]*2,
                         index=index)

        est = build_estimator()

        res = evaluators.check_spacer_across_loci(spacer, loci, est, index = index)

        assert_series_equal(corr, res['score'],
                            check_dtype = False,
                            check_names=False)
        assert_index_equal(corr.index, res.index)


class TestPositionalAgg(object):

    def make_basic_dfs(self):

        target_df = pd.DataFrame([(SeqRecord(Seq('A'*100,
                                             alphabet = Alphabet.generic_dna),
                                         id = 'Seq1'), 10, 110),
                              (SeqRecord(Seq('T'*100,
                                             alphabet = Alphabet.generic_dna),
                                         id = 'Seq2'), 11, 111)],
                              columns = ['Seq', 'Start', 'Stop'])

        spacer_df = pd.DataFrame([(Seq('A'*20, alphabet = Alphabet.generic_rna), 30, 52),
                               (Seq('T'*20, alphabet = Alphabet.generic_rna), 30, 52)],
                              columns = ['spacer', 'Start', 'Stop'])
        est = estimators.CFDEstimator.build_pipeline()

        return target_df, spacer_df, est

    def test_column_names(self):

        target_df, spacer_df, est = self.make_basic_dfs()

        evaluators.positional_aggregation(target_df, spacer_df, est)

        with pytest.raises(AssertionError):
            bseqdf = pd.DataFrame([(Seq('A'*30), 10, 40),
                                   (Seq('T'*30), 11, 41)],
                              columns = ['Sequence', 'Start', 'Stop'])
            evaluators.positional_aggregation(bseqdf, spacer_df, est)

        with pytest.raises(AssertionError):
            bgrnadf = pd.DataFrame([(Seq('A'*20), 10, 15),
                                   (Seq('T'*20), 11, 17)],
                                  columns = ['gRNAs', 'Start', 'Stop'])
            evaluators.positional_aggregation(target_df, bgrnadf, est)

    def test_iterate_overlaps(self):

        target_df = pd.DataFrame([('S1', 50, 200),
                              ('S2', 150, 250),
                              ('S3', 350, 1100),
                              ('S4', 500, 700),
                              ('S5', 400, 900)],
                             columns = ['Seq', 'Start', 'Stop'])

        spacer_df = pd.DataFrame([('g1', 175, 195),
                               ('g2', 600, 620)],
                              columns = ['spacer', 'Start', 'Stop'])

        cors = [('g1',  {'S1', 'S2'}),
                ('g2',  {'S3', 'S4', 'S5'})]

        found = list(evaluators._iterate_grna_seq_overlaps(target_df, spacer_df, 20))
        assert len(found) == 2

        for (grow, sdf), (gkey, cor_seqs) in zip(found, cors):
            assert grow['spacer'] == gkey
            assert set(sdf['Seq']) == cor_seqs

    def make_complicated_dfs(self):

        spacerA = 'A'*10 + 'C'*10
        spacerB = 'T'*10 + 'A'*10

        np.random.seed(0)
        big_seq = make_random_seq(100) + spacerA + 'AGG'
        big_seq += make_random_seq(317) + spacerB + 'TGG' + make_random_seq(97)

        regions = [(50, 250),
                   (110, 300),
                   (200, 350),
                   (300, 500)]
        seqs = []
        correct_calls = []
        num = 0
        for num, (start, stop) in enumerate(regions):
            seqs.append({'Start': start, 'Stop': stop,
                         'Seq': SeqRecord(Seq(big_seq[start:stop], alphabet = Alphabet.generic_dna),
                                          id = str(num)),
                         'Num': num})
            for spacer in [spacerA, spacerB]:
                pos = seqs[-1]['Seq'].seq.find(spacer)
                correct_calls.append({'Num': num,
                                      'spacer': Seq(spacer, alphabet = Alphabet.generic_rna),
                                      'CorPosition': pos if pos >=0 else np.nan,
                                      'IsPresent': pos != -1})

        wrong_seq = 'G'*len(big_seq)
        for num, (start, stop) in enumerate(regions, num):
            seqs.append({'Start': start, 'Stop': stop,
                         'Seq': SeqRecord(Seq(wrong_seq[start:stop], alphabet = Alphabet.generic_dna),
                                          id=str(num+50)),
                         'Num': num})
            for spacer in [spacerA, spacerB]:
                correct_calls.append({'Num': num,
                                      'spacer': Seq(spacer, alphabet = Alphabet.generic_rna),
                                      'CorPosition': np.nan,
                                      'IsPresent': False})

        target_df = pd.DataFrame(seqs)
        spacer_df = pd.DataFrame([{'spacer': Seq(spacerA, alphabet = Alphabet.generic_rna),
                                'Start': 100, 'Stop': 120},
                               {'spacer': Seq(spacerB, alphabet = Alphabet.generic_rna),
                                'Start': 450, 'Stop': 470}])
        cor_df = pd.DataFrame(correct_calls)

        return target_df, spacer_df, cor_df

    def test_positional_aggregation(self):

        target_df, spacer_df, cor_df = self.make_complicated_dfs()
        est = estimators.MismatchEstimator.build_pipeline(miss_tail=0)

        results = evaluators.positional_aggregation(target_df, spacer_df, est, overlap = -1)
        print(results.columns)
        results['StrSpacer'] = results['spacer'].map(str)
        cor_df['StrSpacer'] = cor_df['spacer'].map(str)

        merged = pd.merge(results, cor_df, on = ['Num', 'StrSpacer'], how = 'outer')

        assert merged.loc[merged['IsPresent'].values, 'score'].all()
