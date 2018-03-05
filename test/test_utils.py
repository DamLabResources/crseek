from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq, reverse_complement
from Bio.Alphabet import generic_dna, generic_rna
from crisprtree import utils
from crisprtree import exceptions
import pandas as pd
from pandas.util.testing import assert_series_equal, assert_frame_equal
from tempfile import TemporaryDirectory, NamedTemporaryFile
from Bio import SeqIO
from subprocess import CalledProcessError, check_output
import os
import pytest
import csv
import os
from itertools import product
from tempfile import TemporaryDirectory, NamedTemporaryFile
from unittest.mock import patch
import numpy as np
from Bio.Alphabet import generic_dna, generic_rna
import pandas as pd
import pytest
from Bio import SeqIO
from Bio.Alphabet import generic_alphabet
from Bio.Seq import Seq, reverse_complement
from Bio.SeqRecord import SeqRecord
from pandas.util.testing import assert_series_equal, assert_frame_equal
from crisprtree import utils

formats = ['SeqRecord', 'Seq', 'str', 'tuple']
fmts = list(product(formats, formats)) + [('array', 'str')]


@pytest.mark.parametrize("iofmts", fmts)
def test_smrt_seq_write(iofmts):
    orig_seqs = [SeqRecord(Seq('AAAAAA', alphabet=generic_alphabet),
                           id='A'),
                 SeqRecord(Seq('TTTTT', alphabet=generic_alphabet),
                           id='B'),
                 SeqRecord(Seq('GGGGGG', alphabet=generic_alphabet),
                           id='C')]

    infmt, outfmt = iofmts
    if infmt == 'SeqRecord':
        seqs = [f for f in orig_seqs]
    elif infmt == 'Seq':
        seqs = [f.seq for f in orig_seqs]
    elif infmt == 'str':
        seqs = [str(f.seq) for f in orig_seqs]
    elif infmt == 'array':
        seqs = np.array([str(f.seq) for f in orig_seqs])
    elif infmt == 'tuple':
        seqs = []
        for num, f in enumerate(orig_seqs):
            seqs.append(('S-%i' % num, str(f.seq)))
    else:
        raise AssertionError('Unknown input format %s' % infmt)

    rec_out = list(utils.smrt_seq_convert(outfmt, seqs))
    rec_inp = orig_seqs
    assert len(rec_out) == 3
    for num, (out, inp) in enumerate(zip(rec_out, rec_inp)):
        if infmt == 'SeqRecord':
            inname = inp.id
        elif infmt == 'tuple':
            inname = 'S-%i' % num
        else:
            inname = 'Seq-%i' % num

        if outfmt == 'SeqRecord':
            outname = out.id
            outseq = str(out.seq)
        elif outfmt == 'Seq':
            outname = None
            outseq = str(out)
        elif outfmt == 'str':
            outname = None
            outseq = out
        elif outfmt == 'tuple':
            outname, outseq = out
        else:
            raise AssertionError('Unknown outformat: %s' % outfmt)

        if outname:
            assert outname == inname

        assert outseq == str(inp.seq)


class TestExtract(object):
    def test_basic(self):
        seq = Seq('A' * 20 + 'T' * 20 + 'CCGG' + 'T' * 25 + 'GG', alphabet=generic_dna)

        cor = sorted([Seq('U' * 19 + 'C', alphabet=generic_rna),
                      Seq('U' * 20, alphabet=generic_rna),
                      Seq('A' * 19 + 'C', alphabet=generic_rna),
                      ])

        with pytest.warns(None) as warns:
            res = utils.extract_possible_targets(SeqRecord(seq))

        assert cor == res
        assert len(warns) == 0, 'Still emits BioPython warning!'

    def test_single_strand(self):
        seq = Seq('A' * 20 + 'T' * 20 + 'CCGG' + 'T' * 25 + 'GG', alphabet=generic_dna)

        cor = sorted([Seq('U' * 19 + 'C', alphabet=generic_rna),
                      Seq('U' * 20, alphabet=generic_rna)
                      ])

        res = utils.extract_possible_targets(SeqRecord(seq), both_strands=False)

        assert cor == res

    def test_starts_with_PAM(self):
        seq = Seq('CGG' + 'A' * 20 + 'T' * 20 + 'CCGG' + 'T' * 25 + 'GG', alphabet=generic_dna)

        cor = sorted([Seq('U' * 19 + 'C', alphabet=generic_rna),
                      Seq('U' * 20, alphabet=generic_rna),
                      Seq('A' * 19 + 'C', alphabet=generic_rna)
                      ])

        res = utils.extract_possible_targets(SeqRecord(seq))

        assert cor == res


class TestTiling(object):
    def test_basic(self):

        grna = Seq('A' * 20, alphabet=generic_rna)
        bseq = 'ACTG' * 20
        seqR = SeqRecord(Seq(bseq, alphabet=generic_dna),
                         id='checking')

        res = utils.tile_seqrecord(grna, seqR)

        assert len(res) > 1
        assert (res['spacer'] == grna).all()

        for (name, strand, start), row in res.iterrows():
            assert name == utils._make_record_key(seqR)
            if strand == 1:
                assert str(row['target']) == bseq[start:start + 23]
            else:
                assert str(row['target']) == reverse_complement(bseq[start:start + 23])

    def test_str_spacer(self):

        grna = 'A' * 20
        bseq = 'ACTG' * 20
        seqR = SeqRecord(Seq(bseq, alphabet=generic_dna),
                         id='checking')

        with pytest.raises(ValueError):
            utils.tile_seqrecord(grna, seqR)

    def test_dna_spacer(self):

        grna = Seq('A' * 20, alphabet=generic_dna)
        bseq = 'ACTG' * 20
        seqR = SeqRecord(Seq(bseq, alphabet=generic_dna),
                         id='checking')

        with pytest.raises(exceptions.WrongAlphabetException):
            utils.tile_seqrecord(grna, seqR)

    def test_rna_locus(self):

        grna = Seq('A' * 20, alphabet=generic_rna)
        bseq = 'ACTG' * 20
        seqR = SeqRecord(Seq(bseq, alphabet=generic_rna),
                         id='checking')

        with pytest.raises(exceptions.WrongAlphabetException):
            utils.tile_seqrecord(grna, seqR)


def make_random_seq_restrict(bp):
    """ Utility function for making random sequence
    Parameters
    ----------
    bp : int
        Length of sequence

    Returns
    -------
    str

    """
    return ''.join(np.random.choice(list('AT'), size=bp))


@pytest.mark.skipif(utils._missing_casoffinder(), reason="Need CasOff installed")
class TestCasOff(object):
    def make_basic(self, pam='CGG'):
        np.random.seed(0)
        seqs = [make_random_seq_restrict(50) + 'T' * 20 + pam + make_random_seq_restrict(50),  # hit
                make_random_seq_restrict(50) + 'T' * 19 + 'A' + pam + make_random_seq_restrict(50),  # hit
                make_random_seq_restrict(50) + 'T' * 18 + 'AA' + pam + make_random_seq_restrict(50),  # hit
                make_random_seq_restrict(50) + 'T' * 17 + 'AAA' + pam + make_random_seq_restrict(50),  # hit
                make_random_seq_restrict(50) + 'T' * 14 + 'A' * 6 + pam + make_random_seq_restrict(50),  # no hit
                ]
        seq_recs = [SeqRecord(Seq(s), id='Num-%i' % i, description='') for i, s in enumerate(seqs)]
        spacer = Seq('U' * 20, alphabet=generic_rna)

        cor_index = pd.MultiIndex.from_tuples([('Num-0', 1, 50),
                                               ('Num-1', 1, 50),
                                               ('Num-2', 1, 50),
                                               ('Num-3', 1, 50), ],
                                              names=['name', 'strand', 'left'])
        cor = pd.DataFrame([{'spacer': spacer, 'target': 'T' * 20 + pam},
                            {'spacer': spacer, 'target': 'T' * 19 + 'A' + pam},
                            {'spacer': spacer, 'target': 'T' * 18 + 'AA' + pam},
                            {'spacer': spacer, 'target': 'T' * 17 + 'AAA' + pam}, ],
                           index=cor_index)[['spacer', 'target']]

        return spacer, seq_recs, cor

    def test_basic_seqs(self):
        spacer, seq_recs, cor = self.make_basic()
        res = utils.cas_offinder([spacer], 3, locus=seq_recs)
        assert_frame_equal(res, cor)

    def test_smart_error_for_str_spacers(self):
        _, seq_recs, cor = self.make_basic()
        with pytest.raises(ValueError):
            utils.cas_offinder(['U' * 20], 3, locus=seq_recs)

    def test_smart_error_for_bad_alphabet_spacers(self):
        spacer, seq_recs, cor = self.make_basic()
        spacer.alphabet = generic_dna
        with pytest.raises(exceptions.WrongAlphabetException):
            utils.cas_offinder([spacer], 3, locus=seq_recs)

    def test_smart_error_for_str_locus(self):
        spacer, seq_recs, cor = self.make_basic()
        with pytest.raises(ValueError):
            utils.cas_offinder([spacer], 3, locus=['A' * 500])

    def test_no_hits(self):
        _, seq_recs, cor = self.make_basic()

        np.random.seed(20)
        spacer = Seq(''.join(np.random.choice(list('AUCG'), size=20)),
                     alphabet=generic_rna)

        res = utils.cas_offinder([spacer], 0, locus=seq_recs)
        assert len(res.index) == 0
        assert res.index.names == ['name', 'strand', 'left']
        np.testing.assert_array_equal(res.columns,
                                      ['spacer', 'target'])

    def test_change_pam_long(self):
        NmCas9_pam = 'NNNNGATT'
        spacer, seq_recs, cor = self.make_basic(pam='CGCGGATT')
        res = utils.cas_offinder([spacer], 3, locus=seq_recs, pam=NmCas9_pam)
        assert_frame_equal(res, cor)

    def test_change_pam_short(self):
        FnCas9_pam = 'NG'
        spacer, seq_recs, cor = self.make_basic(pam='CG')
        print(cor)
        res = utils.cas_offinder([spacer], 3, locus=seq_recs, pam=FnCas9_pam)
        print(res)
        assert_frame_equal(res, cor)

    def test_make_record_key(self):
        spacer, seq_recs, cor = self.make_basic()
        for n, seqR in enumerate(seq_recs):
            if n==1:
                seqR.description = seqR.id + ' exon 1'
            elif n==2:
                seqR.description = seqR.id + 'exon1'
            elif n==3:
                seqR.description = seqR.id + '\nvariant 1 exon 2'
        res = utils.cas_offinder([spacer], 3, locus=seq_recs)
        cor.index = cor.index.set_levels([utils._make_record_key(seqR) for seqR in seq_recs], level=0)
        assert_frame_equal(res, cor)


    @patch('subprocess.check_call')
    def test_fails_gracefully(self, mock):
        mock.side_effect = FileNotFoundError

        with pytest.raises(AssertionError):
            spacer, seq_recs, cor = self.make_basic()
            res = utils.cas_offinder([spacer], 3, locus=seq_recs)

    def test_basic_path(self):
        spacer, seq_recs, cor = self.make_basic()
        with TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'seqs.fasta'), 'w') as handle:
                SeqIO.write(seq_recs, handle, 'fasta')

            res = utils.cas_offinder([spacer], 5, direc=tmpdir)
            assert_frame_equal(res, cor)

    def test_missing_both(self):
        with pytest.raises(AssertionError):
            utils.cas_offinder(['T' * 20], 5)

    def test_provide_both(self):
        spacer, seq_recs, cor = self.make_basic()
        with TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'seqs.fasta'), 'w') as handle:
                SeqIO.write(seq_recs[1:], handle, 'fasta')

            res = utils.cas_offinder([spacer], 3, direc=tmpdir, locus=[seq_recs[0]])
            assert_frame_equal(res, cor)


class TestOverlaps(object):
    def make_basic(self, extra_items=None):

        if extra_items is None:
            extra_items = []

        index = pd.MultiIndex.from_tuples([('chr1', 1, 5000),  # hit
                                           ('chr1', 1, 10000),  # miss
                                           ('chr2', 1, 100),  # miss
                                           ('chr3', 1, 1400)]  # hit
                                          + extra_items,
                                          names=['Name', 'Strand', 'Left'])
        hits = pd.DataFrame([{'gRNA': 'T' * 20, 'Seq': 'T' * 20 + 'CGG'}] * len(index),
                            index=index)

        return hits

    def write_basic(self, handle, extra_items=None):

        if extra_items is None:
            extra_items = []

        gene_locs = [('chr1', 4000, 7000),
                     ('chr2', 400, 4000),
                     ('chr3', 1000, 2000)] + extra_items
        writer = csv.writer(handle, delimiter='\t',
                            quoting=csv.QUOTE_NONE,
                            dialect=csv.unix_dialect)
        writer.writerows(gene_locs)

    def test_basic(self):

        hits = self.make_basic()

        with NamedTemporaryFile(mode='w', newline='', buffering=1) as handle:
            self.write_basic(handle)

            res = utils.overlap_regions(hits, handle.name)
            cor = pd.Series([True, False, False, True],
                            index=hits.index, name='Region')

            assert_series_equal(res, cor)

    def test_basic_with_strand(self):

        hits = self.make_basic(extra_items=[('chr3', -1, 1400)])

        with NamedTemporaryFile(mode='w', buffering=1, newline='') as handle:
            self.write_basic(handle)

            res = utils.overlap_regions(hits, handle.name)
            cor = pd.Series([True, False, False, True, True],
                            index=hits.index, name='Region')

            assert_series_equal(res, cor)

    def test_fail_gracefully_on_missing_bed(self):

        with pytest.raises(IOError):
            hits = self.make_basic()
            utils.overlap_regions(hits, '/path/that/doesnt/exist')

    def test_fail_gracefully_on_bad_strand(self):

        hits = self.make_basic(extra_items=[('chr1', '+', 45)])
        with NamedTemporaryFile(mode='w', newline='', buffering=1) as handle:
            self.write_basic(handle)
            with pytest.raises(TypeError):
                utils.overlap_regions(hits, handle.name)

    def test_fail_gracefully_on_bad_left(self):

        hits = self.make_basic(extra_items=[('chr1', '+', '45')])
        with NamedTemporaryFile(mode='w', newline='', buffering=1) as handle:
            self.write_basic(handle)
            with pytest.raises(TypeError):
                utils.overlap_regions(hits, handle.name)
