from bin import crisprbam
from pysam import AlignedSegment
from unittest.mock import patch
from pandas.util.testing import assert_series_equal
import pandas as pd
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
from tempfile import NamedTemporaryFile
import pytest


class TestProcessing(object):

    @patch('pysam.AlignedSegment', autospec = AlignedSegment())
    def test_convert_reads_to_rows(self, mock):

        mock.query_sequence = 'A'*100
        mock.reference_start = 50
        mock.reference_end = 150
        mock.query_name = 'Name'

        res = crisprbam.convert_reads_to_rows(mock)

        cor = pd.Series({'Seq': Seq('A'*100, alphabet = generic_dna),
                         'Start': 50,
                         'Stop': 150},
                        name = 'Name')
        assert_series_equal(res, cor)


    def test_load_grna_file(self):

        df = pd.DataFrame([{'gRNA': 'A'*20,
                           'Chrom': 'HXB2',
                            'Name': 'A',
                           'Start': 100},
                           {'gRNA': 'A'*20,
                           'Chrom': 'HXB2',
                            'Name': 'B',
                           'Start': 120},
                           ])

        with NamedTemporaryFile(suffix='.csv') as handle:
            df.to_csv(handle.name)

            res = crisprbam.load_grna_file(handle.name)

        is_seq = res['gRNA'].map(lambda x: type(x) is Seq)
        assert is_seq.all()

    def test_load_grna_file_ban(self):

        df = pd.DataFrame([{'gRNA': 'A'*20,
                           'Start': 100},
                           {'gRNA': 'A'*20,
                           'Start': 120},
                           ])

        with NamedTemporaryFile(suffix='.csv') as handle:
            df.to_csv(handle.name)

            with pytest.raises(AssertionError):
                crisprbam.load_grna_file(handle.name)