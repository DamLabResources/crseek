from crisprtree import preprocessing
from crisprtree import estimators
from crisprtree import evaluators
from crisprtree import annotators
from sklearn.pipeline import Pipeline
from Bio.Seq import reverse_complement
from Bio.SeqFeature import SeqFeature
import numpy as np
import pandas as pd
from pandas.util.testing import assert_series_equal
import pytest

from test.test_evaluators import build_estimator


class TestSeqFeature(object):

    def test_basic(self):

        feat = annotators._build_target_feature(12, 1, 'A'*20)
        assert type(feat) == type(SeqFeature())
        assert feat.location.start == 12
        assert feat.location.end == 35
        assert feat.location.strand == 1

        assert feat.qualifiers.get('gRNA', None) == 'A'*20
        assert feat.qualifiers.get('On Target Score', None) == 1

    def test_reverse_strand(self):

        feat = annotators._build_target_feature(100, -1, 'A'*20)
        assert type(feat) == type(SeqFeature())
        assert feat.location.start == 100
        assert feat.location.end == 77
        assert feat.location.strand == -1

        assert feat.qualifiers.get('gRNA', None) == 'A'*20
        assert feat.qualifiers.get('On Target Score', None) == 1

    def test_assert_error_on_early_start(self):

        with pytest.raises(AssertionError):
            annotators._build_target_feature(12, -1, 'A'*20)

    def test_value_error_on_bad_strand(self):

        with pytest.raises(ValueError):
            annotators._build_target_feature(12, '-', 'A'*20)

    def test_extra_quals_come_along(self):

        feat = annotators._build_target_feature(12, 1, 'A'*20, extra_quals = {'ExtraQual': 50})
        assert feat.qualifiers.get('ExtraQual', None) == 50