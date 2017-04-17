from crisprtree import preprocessing
from crisprtree import estimators
from crisprtree import evaluators
from sklearn.pipeline import Pipeline
from Bio.Seq import reverse_complement
import numpy as np
import pandas as pd
from pandas.util.testing import assert_series_equal


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
        #seqs += [reverse_complement(seq) for seq in seqs]

        corr = pd.Series([True, True, False])#*2

        est = build_estimator()

        res = evaluators.check_grna_across_seqs(gRNA, seqs, est)

        assert_series_equal(corr, res, check_names=False)


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

        assert_series_equal(corr, res, check_names=False)

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

        assert_series_equal(corr, res, check_names=False)

    def test_carries_series_index(self):

        gRNA = 'A'*20

        seqs = ['T'*10 + gRNA + 'TGG' + 'T'*5,
                'T'*15 + gRNA + 'TGG' + 'T'*5,
                'C'*30,
                'G'*12
                ]
        seqs += [reverse_complement(seq) for seq in seqs]

        index = ['Seq%i' % i for i in range(len(seqs))]
        seqs = pd.Series(seqs, index=index)

        corr = pd.Series([True, True, False, False]*2,
                         index=index)

        est = build_estimator()

        res = evaluators.check_grna_across_seqs(gRNA, seqs, est)

        assert_series_equal(corr, res, check_names=False)






