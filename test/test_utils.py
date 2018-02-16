from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq, reverse_complement
from crisprtree import utils
import pandas as pd
from pandas.util.testing import assert_series_equal, assert_frame_equal
from tempfile import TemporaryDirectory, NamedTemporaryFile
from Bio import SeqIO
from subprocess import CalledProcessError
import os
import pytest
import csv
from unittest.mock import patch


class TestExtract(object):

    def test_basic(self):

        seq = 'A'*20 + 'T'*20 + 'CCGG' + 'T'*25 + 'GG'

        cor = sorted(['T'*19 + 'C',
                      'T'*20,
                      'A'*19 + 'C'
                      ])

        res = utils.extract_possible_targets(SeqRecord(Seq(seq)))

        assert cor == res

    def test_single_strand(self):

        seq = 'A'*20 + 'T'*20 + 'CCGG' + 'T'*25 + 'GG'

        cor = sorted(['T'*19 + 'C',
                      'T'*20,
                      ])

        res = utils.extract_possible_targets(SeqRecord(Seq(seq)), both_strands = False)

        assert cor == res

    def test_starts_with_PAM(self):

        seq = 'CGG' + 'A'*20 + 'T'*20 + 'CCGG' + 'T'*25 + 'GG'

        cor = sorted(['T'*19 + 'C',
                      'T'*20,
                      'A'*19 + 'C'
                      ])

        res = utils.extract_possible_targets(SeqRecord(Seq(seq)))

        assert cor == res


class TestTiling(object):

    def test_basic(self):

        grna = 'A'*20
        bseq = 'ACTG'*20
        seqR = SeqRecord(Seq(bseq), id='checking')

        res = utils.tile_seqrecord(grna, seqR)

        assert len(res) > 1
        assert (res['gRNA'] == grna).all()

        for (name, strand, start), row in res.iterrows():
            assert name == 'checking'
            if strand == 1:
                assert row['Seq'] == bseq[start:start+23]
            else:
                assert row['Seq'] == reverse_complement(bseq[start:start+23])


class TestCasOff(object):

    def make_basic(self):
        seqs = ['A'*50 + 'T'*20 + 'CGG' + 'A'*50,           # hit
                'A'*50 + 'T'*19 + 'A' + 'CGG' + 'A'*50,     # hit
                'A'*50 + 'T'*18 + 'AA' + 'CGG' + 'A'*50,    # hit
                'A'*50 + 'T'*17 + 'AAA' + 'CGG' + 'A'*50,   # hit
                'A'*50 + 'T'*14 + 'A'*6 + 'CGG' + 'A'*50,   # no hit
                ]
        seq_recs = [SeqRecord(Seq(s), id='Num-%i' % i, description='') for i, s in enumerate(seqs)]
        gRNA = 'T'*20

        cor_index = pd.MultiIndex.from_tuples([('Num-0', 1, 50),
                                               ('Num-1', 1, 50),
                                               ('Num-2', 1, 50),
                                               ('Num-3', 1, 50),],
                                              names = ['Name', 'Strand', 'Left'])
        cor = pd.DataFrame([{'gRNA': gRNA, 'Seq': 'T'*20 + 'CGG'},
                            {'gRNA': gRNA, 'Seq': 'T'*19 + 'A' + 'CGG'},
                            {'gRNA': gRNA, 'Seq': 'T'*18 + 'AA' + 'CGG'},
                            {'gRNA': gRNA, 'Seq': 'T'*17 + 'AAA'+ 'CGG'},],
                           index = cor_index)[['gRNA', 'Seq']]

        return gRNA, seq_recs, cor

    @pytest.mark.skip(reason="Need CasOff installed")
    def test_basic_seqs(self):

        gRNA, seq_recs, cor = self.make_basic()
        res = utils.cas_offinder([gRNA], 5, seqs=seq_recs)
        assert_frame_equal(res, cor)

    @pytest.mark.skip(reason="Need CasOff installed")
    @patch('subprocess.check_call')
    def test_fails_gracefully(self, mock):
        mock.side_effect = FileNotFoundError

        with pytest.raises(AssertionError):
            gRNA, seq_recs, cor = self.make_basic()
            res = utils.cas_offinder([gRNA], 5, seqs=seq_recs)

    @pytest.mark.skip(reason="Need CasOff installed")
    def test_basic_path(self):

        gRNA, seq_recs, cor = self.make_basic()
        with TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'seqs.fasta'), 'w') as handle:
                SeqIO.write(seq_recs, handle, 'fasta')

            res = utils.cas_offinder([gRNA], 5, direc=tmpdir)
            assert_frame_equal(res, cor)

    @pytest.mark.skip(reason="Need CasOff installed")
    def test_missing_both(self):

        with pytest.raises(AssertionError):
            utils.cas_offinder(['T'*20], 5)

    @pytest.mark.skip(reason="Need CasOff installed")
    def test_provide_both(self):

        gRNA, seq_recs, cor = self.make_basic()
        with TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'seqs.fasta'), 'w') as handle:
                SeqIO.write(seq_recs[1:], handle, 'fasta')

            res = utils.cas_offinder([gRNA], 5, direc=tmpdir, seqs=[seq_recs[0]])
            assert_frame_equal(res, cor)


class TestOverlaps(object):

    def make_basic(self, extra_items = None):

        if extra_items is None:
            extra_items = []

        index = pd.MultiIndex.from_tuples([('chr1', 1, 5000),   # hit
                                           ('chr1', 1, 10000),  # miss
                                           ('chr2', 1, 100),    # miss
                                           ('chr3', 1, 1400)]   # hit
                                           + extra_items,
                                          names = ['Name', 'Strand', 'Left'])
        hits = pd.DataFrame([{'gRNA': 'T'*20, 'Seq': 'T'*20 + 'CGG'}]*len(index),
                            index = index)

        return hits

    def write_basic(self, handle, extra_items = None):

        if extra_items is None:
            extra_items = []

        gene_locs = [('chr1', 4000, 7000),
                         ('chr2', 400, 4000),
                         ('chr3', 1000, 2000)] + extra_items
        writer = csv.writer(handle, delimiter = '\t',
                            quoting=csv.QUOTE_NONE,
                            dialect = csv.unix_dialect)
        writer.writerows(gene_locs)


    def test_basic(self):

        hits = self.make_basic()

        with NamedTemporaryFile(mode='w', newline = '', buffering = 1) as handle:
            self.write_basic(handle)

            res = utils.overlap_regions(hits, handle.name)
            cor = pd.Series([True, False, False, True],
                            index=hits.index, name = 'Region')

            assert_series_equal(res, cor)

    def test_basic_with_strand(self):

        hits = self.make_basic(extra_items = [('chr3', -1, 1400)])

        with NamedTemporaryFile(mode='w', buffering = 1, newline = '') as handle:
            self.write_basic(handle)

            res = utils.overlap_regions(hits, handle.name)
            cor = pd.Series([True, False, False, True, True],
                            index=hits.index, name = 'Region')

            assert_series_equal(res, cor)

    def test_fail_gracefully_on_missing_bed(self):

        with pytest.raises(IOError):
            hits = self.make_basic()
            utils.overlap_regions(hits, '/path/that/doesnt/exist')

    def test_fail_gracefully_on_bad_strand(self):

        hits = self.make_basic(extra_items = [('chr1', '+', 45)])
        with NamedTemporaryFile(mode='w',newline = '', buffering = 1) as handle:
            self.write_basic(handle)
            with pytest.raises(TypeError):
                utils.overlap_regions(hits, handle.name)

    def test_fail_gracefully_on_bad_left(self):

        hits = self.make_basic(extra_items = [('chr1', '+', '45')])
        with NamedTemporaryFile(mode='w', newline = '', buffering = 1) as handle:
            self.write_basic(handle)
            with pytest.raises(TypeError):
                utils.overlap_regions(hits, handle.name)
